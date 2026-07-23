"""
SSL4EO-S12 MoCo ViT-S/16 + SLR + UNet 的监督分割 Task。

完整论文复现流程中，本 Task 用于第二阶段：

SLR-MAE 自监督 checkpoint
→ 读取 checkpoint["slr_state_dict"]
→ 初始化相同结构的编码器 SLR
→ 加载 SLR 参数
→ 使用带标签数据监督微调
"""

from __future__ import annotations

from collections.abc import Sequence

from torchgeo.trainers import SemanticSegmentationTask

from models.peft_layers import count_parameters
from models.ssl4eo_moco_vit_unet_slr import (
    SSL4EOMoCoViTUNetSLR,
)


class SSL4EOMoCoViTUNetSLRTask(SemanticSegmentationTask):
    """SLR 作物语义分割任务。"""

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        slr_rank: int = 16,
        slr_blocks: Sequence[int] | None = None,
        slr_qkv: bool = True,
        slr_attn_proj: bool = True,
        slr_mlp_fc1: bool = True,
        slr_mlp_fc2: bool = True,
        slr_checkpoint_path: str | None = None,
        strict_slr_loading: bool = True,
        in_channels: int = 13,
        num_classes: int = 6,
        loss: str = "ce",
        class_weights=None,
        ignore_index: int | None = 0,
        lr: float = 1.0e-3,
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

        self.ssl4eo_checkpoint_path = checkpoint_path
        self.ssl4eo_image_size = int(image_size)
        self.ssl4eo_slr_rank = int(slr_rank)
        self.ssl4eo_slr_blocks = (
            None
            if slr_blocks is None
            else tuple(int(index) for index in slr_blocks)
        )
        self.ssl4eo_slr_qkv = bool(slr_qkv)
        self.ssl4eo_slr_attn_proj = bool(slr_attn_proj)
        self.ssl4eo_slr_mlp_fc1 = bool(slr_mlp_fc1)
        self.ssl4eo_slr_mlp_fc2 = bool(slr_mlp_fc2)
        self.ssl4eo_slr_checkpoint_path = slr_checkpoint_path
        self.ssl4eo_strict_slr_loading = bool(strict_slr_loading)

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

        self.hparams["model"] = "ssl4eo_moco_vit_unet_slr"
        self.hparams["backbone"] = (
            "ssl4eo_s12_moco_vit_small_patch16"
        )
        self.hparams["weights"] = checkpoint_path
        self.hparams["image_size"] = image_size
        self.hparams["fine_tuning_method"] = "slr_ssl_ft"
        self.hparams["slr_rank"] = slr_rank
        self.hparams["slr_blocks"] = list(self.model.slr_blocks)
        self.hparams["slr_qkv"] = slr_qkv
        self.hparams["slr_attn_proj"] = slr_attn_proj
        self.hparams["slr_mlp_fc1"] = slr_mlp_fc1
        self.hparams["slr_mlp_fc2"] = slr_mlp_fc2
        self.hparams["slr_checkpoint_path"] = (
            self.model.loaded_slr_checkpoint_path
        )

        for key, value in statistics.items():
            self.hparams[key] = value

        for key, value in self.model.slr_parameter_statistics.items():
            self.hparams[f"vit_{key}"] = value

        self.save_hyperparameters(
            {
                "checkpoint_path": checkpoint_path,
                "image_size": image_size,
                "slr_rank": slr_rank,
                "slr_blocks": list(self.model.slr_blocks),
                "slr_qkv": slr_qkv,
                "slr_attn_proj": slr_attn_proj,
                "slr_mlp_fc1": slr_mlp_fc1,
                "slr_mlp_fc2": slr_mlp_fc2,
                "slr_checkpoint_path": (
                    self.model.loaded_slr_checkpoint_path
                ),
                "strict_slr_loading": strict_slr_loading,
            }
        )

    def configure_models(self) -> None:
        """创建并初始化 SLR ViT-UNet。"""
        self.model = SSL4EOMoCoViTUNetSLR(
            checkpoint_path=self.ssl4eo_checkpoint_path,
            num_classes=int(self.hparams["num_classes"]),
            image_size=self.ssl4eo_image_size,
            slr_rank=self.ssl4eo_slr_rank,
            slr_blocks=self.ssl4eo_slr_blocks,
            slr_qkv=self.ssl4eo_slr_qkv,
            slr_attn_proj=self.ssl4eo_slr_attn_proj,
            slr_mlp_fc1=self.ssl4eo_slr_mlp_fc1,
            slr_mlp_fc2=self.ssl4eo_slr_mlp_fc2,
            slr_checkpoint_path=self.ssl4eo_slr_checkpoint_path,
            strict_slr_loading=self.ssl4eo_strict_slr_loading,
        )

    def on_fit_start(self) -> None:
        """记录 batch size、参数统计与自监督 checkpoint。"""
        datamodule = self.trainer.datamodule
        experiment_config = {
            "batch_size": getattr(datamodule, "batch_size", None),
            "slr_checkpoint_path": (
                self.model.loaded_slr_checkpoint_path
            ),
            **count_parameters(self.model),
        }

        for key, value in experiment_config.items():
            self.hparams[key] = value

        if self.logger is not None:
            self.logger.log_hyperparams(experiment_config)
