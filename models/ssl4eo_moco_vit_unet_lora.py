"""
SSL4EO-S12 MoCo ViT-S/16 + Q/V LoRA + SMP UNet Decoder。
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from models.peft_layers import (
    count_parameters,
    freeze_module,
    inject_lora_into_vit,
)
from models.ssl4eo_moco_vit_unet import SSL4EOMoCoViTUNet


class SSL4EOMoCoViTUNetLoRA(SSL4EOMoCoViTUNet):
    """
    在现有 ViT-UNet 上使用 LoRA。

    冻结部分：
    - SSL4EO-S12 MoCo ViT 原始参数。

    训练部分：
    - 每个指定 Transformer block 的 Query LoRA；
    - 每个指定 Transformer block 的 Value LoRA；
    - feature_norms；
    - 原有多尺度 ScaleAdapter；
    - SMP UNet decoder；
    - segmentation head。

    说明：
    当前模型中的 ``encoder.adapters`` 是空间尺度适配器，
    与 AdaptFormer 无关，在 LoRA 实验中仍然正常训练。
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 6,
        image_size: int = 256,
        lora_rank: int = 4,
        lora_alpha: float = 4.0,
        lora_dropout: float = 0.0,
        lora_blocks: Sequence[int] | None = None,
        lora_query: bool = True,
        lora_value: bool = True,
    ) -> None:
        # 先建立原始模型并冻结完整 ViT。
        super().__init__(
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            image_size=image_size,
            freeze_vit=True,
        )

        # 再次明确冻结 ViT，避免未来基类修改造成意外解冻。
        freeze_module(self.encoder.vit)

        self.lora_blocks = inject_lora_into_vit(
            vit=self.encoder.vit,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            block_indices=lora_blocks,
            enable_query=lora_query,
            enable_value=lora_value,
        )

        self.peft_method = "lora"
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)

        # 注入后的 LoRA 参数默认 requires_grad=True。
        # 原始 ViT 参数保留 requires_grad=False。
        self.parameter_statistics = count_parameters(self)

    def forward(self, x: Tensor) -> Tensor:
        """沿用原始 ViT-UNet 前向传播。"""
        return super().forward(x)
