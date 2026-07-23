from __future__ import annotations

from torchgeo.trainers import SemanticSegmentationTask

from models.peft_layers import count_parameters
from models.ssl4eo_moco_vit_unet_vpt import SSL4EOMoCoViTUNetVPT


class SSL4EOMoCoViTUNetVPTTask(SemanticSegmentationTask):
    """TorchGeo task for SSL4EO-S12 MoCo ViT-S/16 + VPT + UNet."""

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        prompt_length: int = 10,
        vpt_type: str = "deep",
        prompt_dropout: float = 0.0,
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

        self.ssl4eo_checkpoint_path = checkpoint_path
        self.ssl4eo_image_size = int(image_size)
        self.ssl4eo_prompt_length = int(prompt_length)
        self.ssl4eo_vpt_type = str(vpt_type)
        self.ssl4eo_prompt_dropout = float(prompt_dropout)

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

        self.hparams["model"] = "ssl4eo_moco_vit_unet_vpt"
        self.hparams["backbone"] = "ssl4eo_s12_moco_vit_small_patch16"
        self.hparams["weights"] = checkpoint_path
        self.hparams["image_size"] = image_size
        self.hparams["fine_tuning_method"] = f"vpt_{self.model.vpt_type}"
        self.hparams["prompt_length"] = prompt_length
        self.hparams["vpt_type"] = self.model.vpt_type
        self.hparams["prompt_dropout"] = prompt_dropout

        for key, value in statistics.items():
            self.hparams[key] = value

        self.save_hyperparameters(
            {
                "checkpoint_path": checkpoint_path,
                "image_size": image_size,
                "prompt_length": prompt_length,
                "vpt_type": self.model.vpt_type,
                "prompt_dropout": prompt_dropout,
            }
        )

    def configure_models(self) -> None:
        self.model = SSL4EOMoCoViTUNetVPT(
            checkpoint_path=self.ssl4eo_checkpoint_path,
            num_classes=int(self.hparams["num_classes"]),
            image_size=self.ssl4eo_image_size,
            prompt_length=self.ssl4eo_prompt_length,
            vpt_type=self.ssl4eo_vpt_type,
            prompt_dropout=self.ssl4eo_prompt_dropout,
        )

    def on_fit_start(self) -> None:
        datamodule = self.trainer.datamodule
        experiment_config = {
            "batch_size": getattr(datamodule, "batch_size", None),
            **count_parameters(self.model),
        }

        for key, value in experiment_config.items():
            self.hparams[key] = value

        if self.logger is not None:
            self.logger.log_hyperparams(experiment_config)
