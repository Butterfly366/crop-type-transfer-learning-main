"""
SSL4EO-S12 MoCo ViT-S/16 + AdaptFormer + UNet 的 TorchGeo 训练任务。
"""

from __future__ import annotations

from collections.abc import Sequence

from torchgeo.trainers import SemanticSegmentationTask

from models.peft_layers import count_parameters
from models.ssl4eo_moco_vit_unet_adaptformer import (
    SSL4EOMoCoViTUNetAdaptFormer,
)


class SSL4EOMoCoViTUNetAdaptFormerTask(SemanticSegmentationTask):
    """
    AdaptFormer 作物语义分割任务。

    继承 SemanticSegmentationTask 已有的训练、验证、测试、
    损失函数、指标、优化器和学习率调度器逻辑。
    """

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        adapter_bottleneck_dim: int = 64,
        adapter_scale: float = 0.1,
        adapter_dropout: float = 0.0,
        adapter_learnable_scale: bool = False,
        adapter_blocks: Sequence[int] | None = None,
        in_channels: int = 13,
        num_classes: int = 6,
        loss: str = "ce",
        class_weights=None,
        ignore_index: int | None = 0,
        lr: float = 1e-3,
        patience: int = 1,
    ) -> None:
        if in_channels != 13:
            raise ValueError(
                "SSL4EO-S12 MoCo ViT-S/16 要求 13 波段输入，"
                f"当前为 {in_channels}。"
            )

        if image_size != 256:
            raise ValueError(
                "当前多尺度特征适配器按照 256×256 输入设计，"
                f"当前为 {image_size}。"
            )

        # configure_models() 会在父类 __init__ 内调用，
        # 因此自定义模型参数必须提前保存。
        self.ssl4eo_checkpoint_path = checkpoint_path
        self.ssl4eo_image_size = int(image_size)
        self.ssl4eo_adapter_bottleneck_dim = int(adapter_bottleneck_dim)
        self.ssl4eo_adapter_scale = float(adapter_scale)
        self.ssl4eo_adapter_dropout = float(adapter_dropout)
        self.ssl4eo_adapter_learnable_scale = bool(
            adapter_learnable_scale
        )
        self.ssl4eo_adapter_blocks = (
            None
            if adapter_blocks is None
            else tuple(int(index) for index in adapter_blocks)
        )

        super().__init__(
            model="unet",
            backbone="resnet50",
            weights=None,
            in_channels=in_channels,
            num_classes=num_classes,
            num_filters=1,
            loss=loss,
            class_weights=class_weights,
            ignore_index=ignore_index,
            lr=lr,
            patience=patience,
            freeze_backbone=False,
            freeze_decoder=False,
        )

        statistics = count_parameters(self.model)

        self.hparams["model"] = "ssl4eo_moco_vit_unet_adaptformer"
        self.hparams["backbone"] = "ssl4eo_s12_moco_vit_small_patch16"
        self.hparams["weights"] = checkpoint_path
        self.hparams["image_size"] = image_size
        self.hparams["fine_tuning_method"] = "adaptformer"
        self.hparams["adapter_bottleneck_dim"] = adapter_bottleneck_dim
        self.hparams["adapter_scale"] = adapter_scale
        self.hparams["adapter_dropout"] = adapter_dropout
        self.hparams["adapter_learnable_scale"] = (
            adapter_learnable_scale
        )
        self.hparams["adapter_blocks"] = list(self.model.adapter_blocks)

        for key, value in statistics.items():
            self.hparams[key] = value

        self.save_hyperparameters(
            {
                "checkpoint_path": checkpoint_path,
                "image_size": image_size,
                "adapter_bottleneck_dim": adapter_bottleneck_dim,
                "adapter_scale": adapter_scale,
                "adapter_dropout": adapter_dropout,
                "adapter_learnable_scale": adapter_learnable_scale,
                "adapter_blocks": list(self.model.adapter_blocks),
            }
        )

    def configure_models(self) -> None:
        """创建 AdaptFormer ViT-UNet。"""
        self.model = SSL4EOMoCoViTUNetAdaptFormer(
            checkpoint_path=self.ssl4eo_checkpoint_path,
            num_classes=int(self.hparams["num_classes"]),
            image_size=self.ssl4eo_image_size,
            adapter_bottleneck_dim=(
                self.ssl4eo_adapter_bottleneck_dim
            ),
            adapter_scale=self.ssl4eo_adapter_scale,
            adapter_dropout=self.ssl4eo_adapter_dropout,
            adapter_learnable_scale=(
                self.ssl4eo_adapter_learnable_scale
            ),
            adapter_blocks=self.ssl4eo_adapter_blocks,
        )

    def on_fit_start(self) -> None:
        """记录 batch size 和参数统计。"""
        datamodule = self.trainer.datamodule
        experiment_config = {
            "batch_size": getattr(datamodule, "batch_size", None),
            **count_parameters(self.model),
        }

        for key, value in experiment_config.items():
            self.hparams[key] = value

        if self.logger is not None:
            self.logger.log_hyperparams(experiment_config)
