"""
TorchGeo SemanticSegmentationTask：
SSL4EO-S12 MAE ViT-S/16 + GDA-SLR + UNet。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchgeo.trainers import SemanticSegmentationTask

from models.peft_layers import count_parameters
from models.ssl4eo_mae_vit_unet_slr import (
    SSL4EOMAEViTUNetSLR,
)


class SSL4EOMAEViTUNetSLRTask(SemanticSegmentationTask):
    """第二阶段：SLR 与 UNet decoder 联合监督微调。"""

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        selected_blocks: Sequence[int] = (1, 4, 7, 9, 11),
        slr_rank: int = 16,
        slr_blocks: Sequence[int] | None = None,
        patch_embed_adapter: bool = True,
        norm_trainable: bool = True,
        train_cls_token: bool = True,
        slr_checkpoint_path: str | None = None,
        strict_slr_loading: bool = True,
        in_channels: int = 13,
        num_classes: int = 6,
        loss: str = "ce",
        class_weights: list[float] | None = None,
        ignore_index: int | None = 0,
        lr: float = 1.0e-4,
        head_lr: float = 5.0e-4,
        weight_decay: float = 0.05,
        patience: int = 5,
    ) -> None:
        if in_channels != 13:
            raise ValueError(
                "SSL4EO-S12 MAE ViT-S/16 要求 13 波段输入。"
            )
        if image_size != 256:
            raise ValueError(
                "当前 ScaleAdapter/UNet 配置要求 image_size=256。"
            )
        if lr <= 0:
            raise ValueError("lr 必须大于 0。")
        if head_lr <= 0:
            raise ValueError("head_lr 必须大于 0。")
        if weight_decay < 0:
            raise ValueError("weight_decay 不能小于 0。")
        if patience < 0:
            raise ValueError("patience 不能小于 0。")

        # configure_models 会在父类初始化过程中调用，
        # 因此必须提前保存构建模型所需的参数。
        self.ssl4eo_checkpoint_path = checkpoint_path
        self.ssl4eo_image_size = int(image_size)
        self.ssl4eo_selected_blocks = tuple(
            int(index)
            for index in selected_blocks
        )
        self.ssl4eo_slr_rank = int(slr_rank)
        self.ssl4eo_slr_blocks = (
            None
            if slr_blocks is None
            else tuple(int(index) for index in slr_blocks)
        )
        self.ssl4eo_patch_embed_adapter = bool(
            patch_embed_adapter
        )
        self.ssl4eo_norm_trainable = bool(
            norm_trainable
        )
        self.ssl4eo_train_cls_token = bool(
            train_cls_token
        )
        self.ssl4eo_slr_checkpoint_path = (
            slr_checkpoint_path
        )
        self.ssl4eo_strict_slr_loading = bool(
            strict_slr_loading
        )
        self.ssl4eo_head_lr = float(head_lr)
        self.ssl4eo_weight_decay = float(weight_decay)

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

        self.hparams["model"] = (
            "ssl4eo_mae_vit_unet_slr"
        )
        self.hparams["backbone"] = (
            "ssl4eo_s12_mae_vit_small_patch16"
        )
        self.hparams["weights"] = checkpoint_path
        self.hparams["image_size"] = image_size
        self.hparams["selected_blocks"] = list(
            self.ssl4eo_selected_blocks
        )
        self.hparams["fine_tuning_method"] = (
            "slr_ssl_ft"
        )
        self.hparams["slr_rank"] = slr_rank
        self.hparams["slr_blocks"] = list(
            self.model.encoder.slr_blocks
        )
        self.hparams["patch_embed_adapter"] = (
            patch_embed_adapter
        )
        self.hparams["norm_trainable"] = (
            norm_trainable
        )
        self.hparams["train_cls_token"] = (
            train_cls_token
        )
        self.hparams["slr_checkpoint_path"] = (
            self.model.loaded_slr_checkpoint_path
        )
        self.hparams["strict_slr_loading"] = (
            strict_slr_loading
        )
        self.hparams["head_lr"] = head_lr
        self.hparams["weight_decay"] = weight_decay

        for key, value in statistics.items():
            self.hparams[key] = value

        for (
            key,
            value,
        ) in self.model.slr_parameter_statistics.items():
            self.hparams[f"encoder_{key}"] = value

        self.save_hyperparameters(
            {
                "checkpoint_path": checkpoint_path,
                "image_size": image_size,
                "selected_blocks": list(
                    self.ssl4eo_selected_blocks
                ),
                "slr_rank": slr_rank,
                "slr_blocks": list(
                    self.model.encoder.slr_blocks
                ),
                "patch_embed_adapter": (
                    patch_embed_adapter
                ),
                "norm_trainable": norm_trainable,
                "train_cls_token": train_cls_token,
                "slr_checkpoint_path": (
                    self.model.loaded_slr_checkpoint_path
                ),
                "strict_slr_loading": (
                    strict_slr_loading
                ),
                "head_lr": head_lr,
                "weight_decay": weight_decay,
            }
        )

    def configure_models(self) -> None:
        """创建 MAE ViT-S/16 + SLR + UNet。"""

        self.model = SSL4EOMAEViTUNetSLR(
            checkpoint_path=(
                self.ssl4eo_checkpoint_path
            ),
            num_classes=int(
                self.hparams["num_classes"]
            ),
            image_size=self.ssl4eo_image_size,
            selected_blocks=(
                self.ssl4eo_selected_blocks
            ),
            slr_rank=self.ssl4eo_slr_rank,
            slr_blocks=self.ssl4eo_slr_blocks,
            patch_embed_adapter=(
                self.ssl4eo_patch_embed_adapter
            ),
            norm_trainable=(
                self.ssl4eo_norm_trainable
            ),
            train_cls_token=(
                self.ssl4eo_train_cls_token
            ),
            slr_checkpoint_path=(
                self.ssl4eo_slr_checkpoint_path
            ),
            strict_slr_loading=(
                self.ssl4eo_strict_slr_loading
            ),
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """
        使用两组学习率：

        - encoder 中的 SLR、LayerNorm、CLS token：lr
        - ScaleAdapter、UNet decoder、segmentation head：head_lr

        这与 GDA 下游阶段 adapter_lr/head_lr 分离的思路一致。
        """

        encoder_parameters = [
            parameter
            for parameter in self.model.encoder.parameters()
            if parameter.requires_grad
        ]

        head_parameters = [
            parameter
            for module in (
                self.model.decoder,
                self.model.segmentation_head,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        ]

        # feature_norms 与 ScaleAdapter 位于 encoder 对象中，
        # 但它们属于新建任务头侧模块，应使用 head_lr。
        feature_adapter_parameters = [
            parameter
            for module in (
                self.model.encoder.feature_norms,
                self.model.encoder.adapters,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        ]

        feature_adapter_ids = {
            id(parameter)
            for parameter in feature_adapter_parameters
        }

        encoder_parameters = [
            parameter
            for parameter in encoder_parameters
            if id(parameter) not in feature_adapter_ids
        ]

        parameter_groups = []

        if encoder_parameters:
            parameter_groups.append(
                {
                    "params": encoder_parameters,
                    "lr": float(self.hparams["lr"]),
                    "name": "encoder_slr",
                }
            )

        combined_head_parameters = (
            feature_adapter_parameters
            + head_parameters
        )

        if combined_head_parameters:
            parameter_groups.append(
                {
                    "params": combined_head_parameters,
                    "lr": self.ssl4eo_head_lr,
                    "name": "decoder_head",
                }
            )

        if not parameter_groups:
            raise RuntimeError("没有可训练参数。")

        optimizer = AdamW(
            parameter_groups,
            weight_decay=self.ssl4eo_weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=int(self.hparams["patience"]),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.monitor,
                "interval": "epoch",
                "frequency": 1,
                "name": "ssl4eo_mae_slr_plateau",
            },
        }

    def on_fit_start(self) -> None:
        """记录数据批量、参数统计及 SLR checkpoint。"""

        datamodule = self.trainer.datamodule

        experiment_config = {
            "batch_size": getattr(
                datamodule,
                "batch_size",
                None,
            ),
            "slr_checkpoint_path": (
                self.model.loaded_slr_checkpoint_path
            ),
            "adapter_lr": float(
                self.hparams["lr"]
            ),
            "head_lr": self.ssl4eo_head_lr,
            **count_parameters(self.model),
        }

        for key, value in experiment_config.items():
            self.hparams[key] = value

        if self.logger is not None:
            self.logger.log_hyperparams(
                experiment_config
            )