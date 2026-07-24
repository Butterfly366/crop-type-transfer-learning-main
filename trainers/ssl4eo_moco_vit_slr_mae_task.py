from __future__ import annotations
from typing import Any

import torch
from torch import Tensor
from torchgeo.trainers.base import BaseTask

from models.ssl4eo_moco_vit_slr_mae import SSL4EOMoCoViTSLRMAE


class SSL4EOMoCoViTSLRMAETask(BaseTask):
    monitor = "val_loss"
    mode = "min"

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
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        min_lr: float = 0.0,
        warmup_epochs: int = 10,
        patience: int = 10,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr 必须大于 0。")
        if weight_decay < 0 or min_lr < 0 or warmup_epochs < 0 or patience < 0:
            raise ValueError("weight_decay、min_lr、warmup_epochs、patience 不能为负。")

        super().__init__(ignore=None)
        self.hparams["fine_tuning_method"] = "slr_mae"
        self.hparams["backbone"] = "ssl4eo_s12_moco_vit_small_patch16"

    def configure_models(self) -> None:
        self.model = SSL4EOMoCoViTSLRMAE(
            checkpoint_path=self.hparams["checkpoint_path"],
            image_size=int(self.hparams["image_size"]),
            in_channels=int(self.hparams["in_channels"]),
            patch_size=int(self.hparams["patch_size"]),
            mask_ratio=float(self.hparams["mask_ratio"]),
            slr_rank=int(self.hparams["slr_rank"]),
            slr_blocks=self.hparams["slr_blocks"],
            decoder_embed_dim=int(self.hparams["decoder_embed_dim"]),
            decoder_depth=int(self.hparams["decoder_depth"]),
            decoder_num_heads=int(self.hparams["decoder_num_heads"]),
            norm_pix_loss=bool(self.hparams["norm_pix_loss"]),
        )

    @staticmethod
    def extract_images(batch: Any) -> Tensor:
        if isinstance(batch, Tensor):
            images = batch
        elif isinstance(batch, dict):
            images = None
            for key in ("image", "images", "x", "input", "inputs"):
                value = batch.get(key)
                if isinstance(value, Tensor):
                    images = value
                    break
            if images is None:
                candidates = [v for v in batch.values() if isinstance(v, Tensor) and v.ndim == 4]
                if len(candidates) != 1:
                    raise KeyError("无法从 batch 字典中唯一确定影像张量。")
                images = candidates[0]
        elif isinstance(batch, (tuple, list)) and batch:
            images = batch[0]
            if not isinstance(images, Tensor):
                raise TypeError("batch[0] 不是 Tensor。")
        else:
            raise TypeError(f"不支持的 batch 类型：{type(batch).__name__}。")

        if images.ndim != 4:
            raise ValueError(f"影像必须为 [B,C,H,W]，当前为 {tuple(images.shape)}。")
        return images.float()

    def forward(self, images: Tensor):
        return self.model(images)

    def _shared_step(self, batch: Any, stage: str) -> Tensor:
        images = self.extract_images(batch)
        loss, _, mask = self.model(images)
        bs = int(images.shape[0])

        self.log(
            f"{stage}_loss", loss,
            on_step=stage == "train", on_epoch=True,
            prog_bar=True, logger=True, batch_size=bs, sync_dist=True,
        )
        self.log(
            f"{stage}_mask_ratio_actual", mask.float().mean(),
            on_step=False, on_epoch=True,
            prog_bar=False, logger=True, batch_size=bs, sync_dist=True,
        )
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        del batch_idx, dataloader_idx
        return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        del batch_idx, dataloader_idx
        return self._shared_step(batch, "test")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        del batch_idx, dataloader_idx
        return self.model(self.extract_images(batch))

    def configure_optimizers(self) -> dict[str, Any]:
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("没有可训练参数。")

        optimizer = torch.optim.AdamW(
            params,
            lr=float(self.hparams["lr"]),
            weight_decay=float(self.hparams["weight_decay"]),
            betas=(0.9, 0.95),
        )

        trainer = getattr(self, "_trainer", None)
        max_epochs = max(1, int(trainer.max_epochs)) if trainer is not None else 1
        warmup = min(int(self.hparams["warmup_epochs"]), max_epochs)
        base_lr = float(self.hparams["lr"])
        min_lr = float(self.hparams["min_lr"])

        def factor(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            remain = max(1, max_epochs - warmup)
            progress = min(1.0, max(0.0, (epoch - warmup) / remain))
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
            return (min_lr + (base_lr - min_lr) * cosine) / base_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "slr_mae_lr",
            },
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["slr_state_dict"] = self.model.export_encoder_slr_state_dict()
        checkpoint["slr_metadata"] = {
            "slr_rank": int(self.hparams["slr_rank"]),
            "mask_ratio": float(self.hparams["mask_ratio"]),
            "image_size": int(self.hparams["image_size"]),
            "patch_size": int(self.hparams["patch_size"]),
            "in_channels": int(self.hparams["in_channels"]),
        }

    def on_fit_start(self) -> None:
        stats = self.model.slr_statistics()
        for key, value in stats.items():
            self.hparams[key] = value
        if self.logger is not None:
            self.logger.log_hyperparams(stats)
