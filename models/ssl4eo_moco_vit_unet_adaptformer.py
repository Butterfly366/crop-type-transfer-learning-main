"""
SSL4EO-S12 MoCo ViT-S/16 + AdaptFormer + SMP UNet Decoder。
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from models.peft_layers import (
    count_parameters,
    freeze_module,
    inject_adaptformer_into_vit,
)
from models.ssl4eo_moco_vit_unet import SSL4EOMoCoViTUNet


class SSL4EOMoCoViTUNetAdaptFormer(SSL4EOMoCoViTUNet):
    """
    在现有 ViT-UNet 上使用 AdaptFormer。

    每个指定 Transformer block 的原始 MLP 被包装为：

        原始 MLP(x) + scale * Adapter(x)

    其中 Adapter 为：

        384 -> bottleneck_dim -> GELU -> 384

    冻结部分：
    - SSL4EO-S12 MoCo ViT 原始参数，包括 Attention 和原始 MLP。

    训练部分：
    - AdaptFormer Adapter；
    - 可选的可学习 scale；
    - feature_norms；
    - 原有多尺度 ScaleAdapter；
    - SMP UNet decoder；
    - segmentation head。

    注意：
    ``encoder.adapters`` 是原模型的空间尺度适配器；
    ``block.mlp.adapter`` 才是本文件新增的 AdaptFormer。
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 6,
        image_size: int = 256,
        adapter_bottleneck_dim: int = 64,
        adapter_scale: float = 0.1,
        adapter_dropout: float = 0.0,
        adapter_learnable_scale: bool = False,
        adapter_blocks: Sequence[int] | None = None,
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

        self.adapter_blocks = inject_adaptformer_into_vit(
            vit=self.encoder.vit,
            bottleneck_dim=adapter_bottleneck_dim,
            scale=adapter_scale,
            dropout=adapter_dropout,
            learnable_scale=adapter_learnable_scale,
            block_indices=adapter_blocks,
        )

        self.peft_method = "adaptformer"
        self.adapter_bottleneck_dim = int(adapter_bottleneck_dim)
        self.adapter_scale = float(adapter_scale)
        self.adapter_dropout = float(adapter_dropout)
        self.adapter_learnable_scale = bool(adapter_learnable_scale)

        self.parameter_statistics = count_parameters(self)

    def forward(self, x: Tensor) -> Tensor:
        """沿用原始 ViT-UNet 前向传播。"""
        return super().forward(x)
