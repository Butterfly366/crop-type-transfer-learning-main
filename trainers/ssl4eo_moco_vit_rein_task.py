"""
SSL4EO-S12 MoCo ViT-S/16 + LoRA-REIN + UNet 的 TorchGeo 监督分割 Task。
"""

from __future__ import annotations

from collections.abc import Sequence

from torchgeo.trainers import SemanticSegmentationTask

from models.peft_layers import count_parameters
from models.ssl4eo_moco_vit_unet_rein import (
    SSL4EOMoCoViTUNetREIN,
)


class SSL4EOMoCoViTUNetREINTask(SemanticSegmentationTask):
    """LoRA-REIN 作物语义分割任务。"""

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        rein_type: str = "lora_rein",
        token_length: int = 100,
        lora_dim: int = 16,
        use_softmax: bool = True,
        scale_init: float = 0.001,
        selected_blocks: Sequence[int] = (1, 4, 7, 9, 11),
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
        self.ssl4eo_rein_type = rein_type
        self.ssl4eo_token_length = int(token_length)
        self.ssl4eo_lora_dim = int(lora_dim)
        self.ssl4eo_use_softmax = bool(use_softmax)
        self.ssl4eo_scale_init = float(scale_init)
        self.ssl4eo_selected_blocks = tuple(
            int(index) for index in selected_blocks
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

        self.hparams["model"] = "ssl4eo_moco_vit_unet_rein"
        self.hparams["backbone"] = (
            "ssl4eo_s12_moco_vit_small_patch16"
        )
        self.hparams["weights"] = checkpoint_path
        self.hparams["image_size"] = image_size
        self.hparams["fine_tuning_method"] = "lora_rein"
        self.hparams["rein_type"] = self.model.rein_type
        self.hparams["token_length"] = token_length
        self.hparams["lora_dim"] = lora_dim
        self.hparams["use_softmax"] = use_softmax
        self.hparams["scale_init"] = scale_init
        self.hparams["selected_blocks"] = list(
            self.model.selected_blocks
        )

        for key, value in statistics.items():
            self.hparams[key] = value

        for key, value in (
            self.model.rein_parameter_statistics.items()
        ):
            self.hparams[f"rein_{key}"] = value

        self.save_hyperparameters(
            {
                "checkpoint_path": checkpoint_path,
                "image_size": image_size,
                "rein_type": rein_type,
                "token_length": token_length,
                "lora_dim": lora_dim,
                "use_softmax": use_softmax,
                "scale_init": scale_init,
                "selected_blocks": list(
                    self.model.selected_blocks
                ),
            }
        )

    def configure_models(self) -> None:
        """创建 REIN ViT-UNet。"""
        self.model = SSL4EOMoCoViTUNetREIN(
            checkpoint_path=self.ssl4eo_checkpoint_path,
            num_classes=int(self.hparams["num_classes"]),
            image_size=self.ssl4eo_image_size,
            rein_type=self.ssl4eo_rein_type,
            token_length=self.ssl4eo_token_length,
            lora_dim=self.ssl4eo_lora_dim,
            use_softmax=self.ssl4eo_use_softmax,
            scale_init=self.ssl4eo_scale_init,
            selected_blocks=self.ssl4eo_selected_blocks,
        )

    def on_fit_start(self) -> None:
        """记录 batch size 和参数统计。"""
        datamodule = self.trainer.datamodule

        experiment_config = {
            "batch_size": getattr(
                datamodule,
                "batch_size",
                None,
            ),
            **count_parameters(self.model),
            **{
                f"rein_{key}": value
                for key, value
                in self.model.rein_parameter_statistics.items()
            },
        }

        for key, value in experiment_config.items():
            self.hparams[key] = value

        if self.logger is not None:
            self.logger.log_hyperparams(
                experiment_config
            )
