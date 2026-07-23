"""
Lightning Task for SSL4EO-S12 MoCo ViT-S/16 + SLR-MAE.

该 Task 复用现有作物数据 DataModule，但训练时只读取影像并忽略标签。
支持常见 batch 结构：

- {"image": tensor, "mask": label}
- {"images": tensor, ...}
- {"x": tensor, ...}
- (images, labels)
- [images, labels]
- 直接传入 Tensor

Lightning checkpoint 会额外写入：

    checkpoint["slr_state_dict"]

其中只包含编码器 ViT 的 SLR 参数，可直接交给监督分割模型加载。
"""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch import LightningModule
from torch import Tensor

from models.ssl4eo_moco_vit_slr_mae import (
    SSL4EOMoCoViTSLRMAE,
)


class SSL4EOMoCoViTSLRMAETask(LightningModule):
    """SLR-MAE 自监督训练 Task。"""

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        in_channels: int = 13,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        slr_rank: int = 16,
        slr_blocks: list[int] | None = None,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_pix_loss: bool = False,
        lr: float = 1.0e-4,
        weight_decay: float = 0.05,
        min_lr: float = 0.0,
        warmup_epochs: int = 10,
    ) -> None:
        super().__init__()

        if lr <= 0:
            raise ValueError("lr 必须大于 0。")
        if weight_decay < 0:
            raise ValueError("weight_decay 不能小于 0。")
        if min_lr < 0:
            raise ValueError("min_lr 不能小于 0。")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs 不能小于 0。")

        self.save_hyperparameters()

        self.model = SSL4EOMoCoViTSLRMAE(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            in_channels=in_channels,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            slr_rank=slr_rank,
            slr_blocks=slr_blocks,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            norm_pix_loss=norm_pix_loss,
        )

    @staticmethod
    def extract_images(batch: Any) -> Tensor:
        """从 TorchGeo/普通 PyTorch batch 中提取影像。"""
        if isinstance(batch, Tensor):
            images = batch

        elif isinstance(batch, dict):
            images = None

            for key in (
                "image",
                "images",
                "x",
                "input",
                "inputs",
            ):
                value = batch.get(key)

                if isinstance(value, Tensor):
                    images = value
                    break

            if images is None:
                tensor_items = [
                    value
                    for value in batch.values()
                    if isinstance(value, Tensor)
                    and value.ndim == 4
                ]

                if len(tensor_items) == 1:
                    images = tensor_items[0]
                else:
                    raise KeyError(
                        "无法从 batch 字典中唯一确定影像张量。"
                    )

        elif isinstance(batch, (tuple, list)):
            if not batch:
                raise ValueError("batch 不能为空。")

            images = batch[0]

            if not isinstance(images, Tensor):
                raise TypeError(
                    "batch[0] 不是 Tensor。"
                )

        else:
            raise TypeError(
                "不支持的 batch 类型："
                f"{type(batch).__name__}。"
            )

        if images.ndim != 4:
            raise ValueError(
                f"影像必须为 [B,C,H,W]，当前为 {tuple(images.shape)}。"
            )

        return images.float()

    def forward(
        self,
        images: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.model(images)

    def _shared_step(
        self,
        batch: Any,
        stage: str,
    ) -> Tensor:
        images = self.extract_images(batch)
        loss, _, mask = self.model(images)

        batch_size = int(images.shape[0])

        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        self.log(
            f"{stage}_mask_ratio_actual",
            mask.float().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        return loss

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> Tensor:
        del batch_idx
        return self._shared_step(
            batch,
            stage="train",
        )

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> Tensor:
        del batch_idx
        return self._shared_step(
            batch,
            stage="val",
        )

    def test_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> Tensor:
        del batch_idx
        return self._shared_step(
            batch,
            stage="test",
        )

    def configure_optimizers(self) -> dict[str, Any]:
        trainable_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

        if not trainable_parameters:
            raise RuntimeError("没有可训练参数。")

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
            betas=(0.9, 0.95),
        )

        # 优先使用 Trainer.max_epochs；在模型脱离 Trainer 的单元测试中回退为 1。
        max_epochs = 1

        if self.trainer is not None:
            max_epochs = max(
                1,
                int(self.trainer.max_epochs),
            )

        warmup_epochs = min(
            int(self.hparams.warmup_epochs),
            max_epochs,
        )

        base_lr = float(self.hparams.lr)
        min_lr = float(self.hparams.min_lr)

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)

            remaining = max(
                1,
                max_epochs - warmup_epochs,
            )
            progress = min(
                1.0,
                max(
                    0.0,
                    (epoch - warmup_epochs) / remaining,
                ),
            )

            cosine = 0.5 * (
                1.0 + torch.cos(
                    torch.tensor(
                        progress * torch.pi
                    )
                ).item()
            )

            if base_lr == 0:
                return 1.0

            return (
                min_lr
                + (base_lr - min_lr) * cosine
            ) / base_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "slr_mae_lr",
            },
        }

    def on_save_checkpoint(
        self,
        checkpoint: dict[str, Any],
    ) -> None:
        """
        在 Lightning checkpoint 中额外写入编码器 SLR-only 权重。

        监督模型可直接把整个 .ckpt 路径传给：

            slr_checkpoint_path
        """
        checkpoint["slr_state_dict"] = (
            self.model.export_encoder_slr_state_dict()
        )
        checkpoint["slr_metadata"] = {
            "slr_rank": int(self.hparams.slr_rank),
            "mask_ratio": float(self.hparams.mask_ratio),
            "image_size": int(self.hparams.image_size),
            "patch_size": int(self.hparams.patch_size),
            "in_channels": int(self.hparams.in_channels),
        }

    def on_fit_start(self) -> None:
        statistics = self.model.slr_statistics()

        for key, value in statistics.items():
            self.log(
                f"model_{key}",
                float(value),
                on_step=False,
                on_epoch=False,
                logger=True,
                rank_zero_only=True,
            )
