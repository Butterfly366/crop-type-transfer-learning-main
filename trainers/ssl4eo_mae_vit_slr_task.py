"""
Lightning Task：SSL4EO-S12 MAE ViT-S/16 + GDA SLR 自监督域适配。
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torchgeo.trainers import BaseTask

from models.ssl4eo_mae_vit_slr import SSL4EOMAEViTSLR


class SSL4EOMAEViTSLRTask(BaseTask):
    """训练 SSL4EO-S12 MAE ViT-S/16 中的 GDA SLR 参数。"""

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
        patch_embed_adapter: bool = True,
        norm_trainable: bool = True,
        train_cls_mask_tokens: bool = True,
        loss_on_all_patches: bool = True,
        norm_pix_loss: bool = False,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        lr: float = 1.0e-4,
        weight_decay: float = 0.05,
        min_lr: float = 0.0,
        warmup_epochs: int = 5,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr 必须大于 0。")
        if weight_decay < 0:
            raise ValueError("weight_decay 不能小于 0。")
        if min_lr < 0:
            raise ValueError("min_lr 不能小于 0。")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs 不能小于 0。")

        # BaseTask.__init__ 会依次执行：
        # save_hyperparameters -> configure_models
        # 因此必须先保存模型构建所需的非 hparams 属性。
        self.ssl4eo_checkpoint_path = checkpoint_path
        self.ssl4eo_image_size = int(image_size)
        self.ssl4eo_in_channels = int(in_channels)
        self.ssl4eo_patch_size = int(patch_size)
        self.ssl4eo_mask_ratio = float(mask_ratio)
        self.ssl4eo_slr_rank = int(slr_rank)
        self.ssl4eo_slr_blocks = (
            None
            if slr_blocks is None
            else tuple(int(index) for index in slr_blocks)
        )
        self.ssl4eo_patch_embed_adapter = bool(
            patch_embed_adapter
        )
        self.ssl4eo_norm_trainable = bool(norm_trainable)
        self.ssl4eo_train_cls_mask_tokens = bool(
            train_cls_mask_tokens
        )
        self.ssl4eo_loss_on_all_patches = bool(
            loss_on_all_patches
        )
        self.ssl4eo_norm_pix_loss = bool(norm_pix_loss)
        self.ssl4eo_decoder_embed_dim = int(
            decoder_embed_dim
        )
        self.ssl4eo_decoder_depth = int(decoder_depth)
        self.ssl4eo_decoder_num_heads = int(
            decoder_num_heads
        )

        # 必须继承 TorchGeo BaseTask，才能通过
        # python -m torchgeo fit 的 model 类型检查。
        super().__init__()

    def configure_models(self) -> None:
        """创建 SSL4EO MAE ViT-S/16 + GDA-SLR 模型。"""

        self.model = SSL4EOMAEViTSLR(
            checkpoint_path=self.ssl4eo_checkpoint_path,
            image_size=self.ssl4eo_image_size,
            in_channels=self.ssl4eo_in_channels,
            patch_size=self.ssl4eo_patch_size,
            mask_ratio=self.ssl4eo_mask_ratio,
            slr_rank=self.ssl4eo_slr_rank,
            slr_blocks=self.ssl4eo_slr_blocks,
            patch_embed_adapter=(
                self.ssl4eo_patch_embed_adapter
            ),
            norm_trainable=self.ssl4eo_norm_trainable,
            train_cls_mask_tokens=(
                self.ssl4eo_train_cls_mask_tokens
            ),
            loss_on_all_patches=(
                self.ssl4eo_loss_on_all_patches
            ),
            norm_pix_loss=self.ssl4eo_norm_pix_loss,
            decoder_embed_dim=(
                self.ssl4eo_decoder_embed_dim
            ),
            decoder_depth=self.ssl4eo_decoder_depth,
            decoder_num_heads=(
                self.ssl4eo_decoder_num_heads
            ),
        )

    @staticmethod
    def extract_images(batch: Any) -> Tensor:
        """从 TorchGeo 或普通 PyTorch batch 中提取影像。"""

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
                candidates = [
                    value
                    for value in batch.values()
                    if isinstance(value, Tensor)
                    and value.ndim == 4
                ]

                if len(candidates) != 1:
                    raise KeyError(
                        "无法从 batch 字典中唯一确定影像张量。"
                    )

                images = candidates[0]

        elif isinstance(batch, (tuple, list)):
            if not batch:
                raise ValueError("batch 不能为空。")

            images = batch[0]

            if not isinstance(images, Tensor):
                raise TypeError("batch[0] 不是 Tensor。")

        else:
            raise TypeError(
                "不支持的 batch 类型："
                f"{type(batch).__name__}。"
            )

        if images.ndim != 4:
            raise ValueError(
                "影像必须为 [B,C,H,W]，"
                f"当前为 {tuple(images.shape)}。"
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
        return self._shared_step(batch, "train")

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        del batch_idx, dataloader_idx
        return self._shared_step(batch, "val")

    def test_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        del batch_idx, dataloader_idx
        return self._shared_step(batch, "test")

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del batch_idx, dataloader_idx
        return self.model(self.extract_images(batch))

    def configure_optimizers(self) -> dict[str, Any]:
        """AdamW + warmup/cosine 学习率。"""

        trainable_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

        if not trainable_parameters:
            raise RuntimeError("没有可训练参数。")

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(self.hparams["lr"]),
            weight_decay=float(
                self.hparams["weight_decay"]
            ),
            betas=(0.9, 0.95),
        )

        trainer = getattr(self, "_trainer", None)
        total_steps = 1

        if trainer is not None:
            estimated = int(
                getattr(
                    trainer,
                    "estimated_stepping_batches",
                    0,
                )
                or 0
            )
            max_steps = int(
                getattr(trainer, "max_steps", -1)
            )

            if max_steps > 0:
                total_steps = max_steps
            elif estimated > 0:
                total_steps = estimated

        steps_per_epoch = 1

        if trainer is not None:
            train_batches = getattr(
                trainer,
                "num_training_batches",
                0,
            )

            if isinstance(train_batches, int):
                steps_per_epoch = max(1, train_batches)

        warmup_steps = min(
            total_steps,
            int(self.hparams["warmup_epochs"])
            * steps_per_epoch,
        )
        base_lr = float(self.hparams["lr"])
        min_lr = float(self.hparams["min_lr"])

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(
                    warmup_steps
                )

            remaining = max(
                1,
                total_steps - warmup_steps,
            )
            progress = min(
                1.0,
                max(
                    0.0,
                    (step - warmup_steps) / remaining,
                ),
            )
            cosine = 0.5 * (
                1.0
                + math.cos(math.pi * progress)
            )
            current_lr = (
                min_lr
                + (base_lr - min_lr) * cosine
            )
            return current_lr / base_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "ssl4eo_mae_slr_lr",
            },
        }

    def on_save_checkpoint(
        self,
        checkpoint: dict[str, Any],
    ) -> None:
        """额外保存 SLR-only 权重，供下游监督阶段加载。"""

        checkpoint["slr_state_dict"] = (
            self.model.export_slr_state_dict()
        )
        checkpoint["slr_metadata"] = {
            "backbone": (
                "ssl4eo_s12_mae_vit_small_patch16"
            ),
            "slr_rank": int(
                self.hparams["slr_rank"]
            ),
            "mask_ratio": float(
                self.hparams["mask_ratio"]
            ),
            "image_size": int(
                self.hparams["image_size"]
            ),
            "patch_size": int(
                self.hparams["patch_size"]
            ),
            "in_channels": int(
                self.hparams["in_channels"]
            ),
            "patch_embed_adapter": bool(
                self.hparams["patch_embed_adapter"]
            ),
            "norm_trainable": bool(
                self.hparams["norm_trainable"]
            ),
            "train_cls_mask_tokens": bool(
                self.hparams["train_cls_mask_tokens"]
            ),
            "loss_on_all_patches": bool(
                self.hparams["loss_on_all_patches"]
            ),
        }

    def on_fit_start(self) -> None:
        """将参数统计写入日志。"""

        statistics = self.model.slr_statistics()

        for key, value in statistics.items():
            self.hparams[key] = value

        if self.logger is not None:
            self.logger.log_hyperparams(statistics)
