"""
SSL4EO-S12 MoCo ViT-S/16 + SLR 的 MAE 自监督模型。

实现流程：
1. 加载 SSL4EO-S12 MoCo ViT-S/16；
2. 冻结原始预训练参数；
3. 在编码器 qkv、attn.proj、mlp.fc1、mlp.fc2 中注入 SLR；
4. 随机遮挡 patch；
5. 使用 MAE decoder 重建全部 13 波段 patch；
6. 只在被遮挡 patch 上计算 MSE；
7. 导出编码器 SLR-only 参数，供后续分割模型加载。

为贴近论文官方实现：
- SLR rank 默认 16；
- patch embedding 默认不注入 SLR；
- Encoder/Decoder LayerNorm 可训练；
- MAE decoder 的注意力和 MLP 线性层也采用 SLR；
- decoder_embed 与 decoder_pred 也采用 SLR；
- 所有被包装的原始 Linear 权重冻结。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from timm.models.vision_transformer import Block
from torch import Tensor, nn

from models.slr_layers import (
    ScaledLowRankLinear,
    count_slr_parameters,
    extract_slr_state_dict,
    inject_slr_into_vit,
)
from models.ssl4eo_moco_vit_unet import SSL4EOMoCoViTS16Encoder


def _sincos_1d(
    embed_dim: int,
    positions: np.ndarray,
) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("一维正余弦位置编码维度必须为偶数。")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    output = np.einsum("m,d->md", positions.reshape(-1), omega)

    return np.concatenate(
        [np.sin(output), np.cos(output)],
        axis=1,
    )


def build_2d_sincos_position_embedding(
    embed_dim: int,
    grid_size: int,
    include_cls_token: bool = True,
) -> Tensor:
    """建立固定二维正余弦位置编码。"""
    if embed_dim % 4 != 0:
        raise ValueError("二维位置编码维度必须能够被 4 整除。")

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, -1)

    embedding_h = _sincos_1d(embed_dim // 2, grid[0])
    embedding_w = _sincos_1d(embed_dim // 2, grid[1])
    embedding = np.concatenate(
        [embedding_h, embedding_w],
        axis=1,
    )

    if include_cls_token:
        embedding = np.concatenate(
            [
                np.zeros((1, embed_dim), dtype=np.float32),
                embedding.astype(np.float32),
            ],
            axis=0,
        )

    return torch.from_numpy(embedding).unsqueeze(0)


def unfreeze_layer_norms(module: nn.Module) -> None:
    """使所有 LayerNorm 参数可训练，对齐官方 norm_trainable=True。"""
    for child in module.modules():
        if isinstance(child, nn.LayerNorm):
            for parameter in child.parameters():
                parameter.requires_grad = True


def inject_slr_into_decoder_blocks(
    blocks: nn.ModuleList,
    rank: int,
) -> None:
    """向 MAE decoder 的 Transformer blocks 注入 SLR。"""
    wrapper = nn.Module()
    wrapper.blocks = blocks

    inject_slr_into_vit(
        vit=wrapper,
        rank=rank,
        block_indices=None,
        adapt_qkv=True,
        adapt_attn_proj=True,
        adapt_mlp_fc1=True,
        adapt_mlp_fc2=True,
    )


class SSL4EOMoCoViTSLRMAE(nn.Module):
    """SSL4EO ViT-S/16 + SLR 的 masked autoencoder。"""

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        in_channels: int = 13,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        slr_rank: int = 16,
        slr_blocks: Sequence[int] | None = None,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_pix_loss: bool = False,
    ) -> None:
        super().__init__()

        if image_size != 256:
            raise ValueError(
                "当前 SSL4EO-S12 流程要求 image_size=256。"
            )
        if in_channels != 13:
            raise ValueError(
                "SSL4EO-S12 MoCo ViT-S/16 要求 13 波段输入。"
            )
        if patch_size != 16:
            raise ValueError(
                "当前预训练 ViT 的 patch_size 必须为 16。"
            )
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio 必须位于 (0, 1)。")
        if decoder_embed_dim % decoder_num_heads != 0:
            raise ValueError(
                "decoder_embed_dim 必须能被 decoder_num_heads 整除。"
            )

        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size**2
        self.mask_ratio = float(mask_ratio)
        self.slr_rank = int(slr_rank)
        self.norm_pix_loss = bool(norm_pix_loss)

        # 借用项目中已经验证过的 SSL4EO 权重加载与位置编码插值逻辑。
        loaded_encoder = SSL4EOMoCoViTS16Encoder(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            selected_blocks=(1, 4, 7, 9, 11),
            freeze_vit=True,
        )
        self.vit = loaded_encoder.vit

        # 原始 ViT 全部冻结。
        for parameter in self.vit.parameters():
            parameter.requires_grad = False

        self.slr_blocks = inject_slr_into_vit(
            vit=self.vit,
            rank=self.slr_rank,
            block_indices=slr_blocks,
            adapt_qkv=True,
            adapt_attn_proj=True,
            adapt_mlp_fc1=True,
            adapt_mlp_fc2=True,
        )

        # 官方默认允许 norm 训练。
        unfreeze_layer_norms(self.vit)

        encoder_embed_dim = int(self.vit.embed_dim)

        # MAE decoder。
        self.decoder_embed = ScaledLowRankLinear(
            nn.Linear(
                encoder_embed_dim,
                decoder_embed_dim,
                bias=True,
            ),
            rank=self.slr_rank,
        )

        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, decoder_embed_dim)
        )

        decoder_position = build_2d_sincos_position_embedding(
            embed_dim=decoder_embed_dim,
            grid_size=self.grid_size,
            include_cls_token=True,
        )
        self.register_buffer(
            "decoder_pos_embed",
            decoder_position,
            persistent=True,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(decoder_depth)
            ]
        )

        # 冻结 decoder 原始 Linear，再注入 SLR。
        for block in self.decoder_blocks:
            for parameter in block.parameters():
                parameter.requires_grad = False

        inject_slr_into_decoder_blocks(
            self.decoder_blocks,
            rank=self.slr_rank,
        )
        unfreeze_layer_norms(self.decoder_blocks)

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        patch_vector_dim = (
            self.patch_size
            * self.patch_size
            * self.in_channels
        )
        self.decoder_pred = ScaledLowRankLinear(
            nn.Linear(
                decoder_embed_dim,
                patch_vector_dim,
                bias=True,
            ),
            rank=self.slr_rank,
        )

        self._initialize_mae_parameters()

    def _initialize_mae_parameters(self) -> None:
        """初始化 mask token；SLR 层自身已完成官方初始化。"""
        nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, images: Tensor) -> Tensor:
        """[B,C,H,W] -> [B,L,p*p*C]。"""
        p = self.patch_size

        if images.ndim != 4:
            raise ValueError("images 必须是四维张量。")
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"输入波段数错误：{images.shape[1]}。"
            )
        if images.shape[2] != images.shape[3]:
            raise ValueError("输入影像必须为正方形。")
        if images.shape[2] != self.image_size:
            raise ValueError(
                f"输入尺寸必须为 {self.image_size}。"
            )

        batch_size, channels, height, width = images.shape
        grid_h = height // p
        grid_w = width // p

        patches = images.reshape(
            batch_size,
            channels,
            grid_h,
            p,
            grid_w,
            p,
        )
        patches = torch.einsum(
            "nchpwq->nhwpqc",
            patches,
        )
        return patches.reshape(
            batch_size,
            grid_h * grid_w,
            p * p * channels,
        )

    def unpatchify(self, patches: Tensor) -> Tensor:
        """[B,L,p*p*C] -> [B,C,H,W]。"""
        p = self.patch_size
        grid_size = int(patches.shape[1] ** 0.5)

        if grid_size * grid_size != patches.shape[1]:
            raise ValueError("patch 数量不能构成正方形网格。")

        images = patches.reshape(
            patches.shape[0],
            grid_size,
            grid_size,
            p,
            p,
            self.in_channels,
        )
        images = torch.einsum(
            "nhwpqc->nchpwq",
            images,
        )
        return images.reshape(
            patches.shape[0],
            self.in_channels,
            grid_size * p,
            grid_size * p,
        )

    @staticmethod
    def random_masking(
        tokens: Tensor,
        mask_ratio: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        对 patch token 做逐样本随机遮挡。

        返回：
        - 可见 token；
        - 二值 mask，0=保留、1=遮挡；
        - ids_restore，用于恢复原 patch 顺序。
        """
        batch_size, length, channels = tokens.shape
        keep_length = int(length * (1.0 - mask_ratio))

        noise = torch.rand(
            batch_size,
            length,
            device=tokens.device,
        )

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep_length]

        visible = torch.gather(
            tokens,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(
                -1,
                -1,
                channels,
            ),
        )

        mask = torch.ones(
            batch_size,
            length,
            device=tokens.device,
        )
        mask[:, :keep_length] = 0
        mask = torch.gather(
            mask,
            dim=1,
            index=ids_restore,
        )

        return visible, mask, ids_restore

    def forward_encoder(
        self,
        images: Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """执行 patch embedding、随机遮挡和 SLR ViT 编码。"""
        ratio = (
            self.mask_ratio
            if mask_ratio is None
            else float(mask_ratio)
        )

        tokens = self.vit.patch_embed(images)

        # 使用已经由项目加载逻辑插值到 16×16 的预训练位置编码。
        tokens = tokens + self.vit.pos_embed[:, 1:, :]

        visible, mask, ids_restore = self.random_masking(
            tokens,
            ratio,
        )

        cls_token = (
            self.vit.cls_token
            + self.vit.pos_embed[:, :1, :]
        )
        cls_tokens = cls_token.expand(
            images.shape[0],
            -1,
            -1,
        )

        encoded = torch.cat(
            [cls_tokens, visible],
            dim=1,
        )

        pos_drop = getattr(
            self.vit,
            "pos_drop",
            None,
        )
        if pos_drop is not None:
            encoded = pos_drop(encoded)

        patch_drop = getattr(
            self.vit,
            "patch_drop",
            None,
        )
        if patch_drop is not None:
            encoded = patch_drop(encoded)

        norm_pre = getattr(
            self.vit,
            "norm_pre",
            None,
        )
        if norm_pre is not None:
            encoded = norm_pre(encoded)

        for block in self.vit.blocks:
            encoded = block(encoded)

        encoded = self.vit.norm(encoded)

        return encoded, mask, ids_restore

    def forward_decoder(
        self,
        encoded: Tensor,
        ids_restore: Tensor,
    ) -> Tensor:
        """恢复 mask token，并预测每个 patch 的 13 波段像素。"""
        decoded = self.decoder_embed(encoded)

        visible_patch_tokens = decoded[:, 1:, :]
        num_mask_tokens = (
            ids_restore.shape[1]
            - visible_patch_tokens.shape[1]
        )

        mask_tokens = self.mask_token.repeat(
            decoded.shape[0],
            num_mask_tokens,
            1,
        )

        restored = torch.cat(
            [visible_patch_tokens, mask_tokens],
            dim=1,
        )
        restored = torch.gather(
            restored,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(
                -1,
                -1,
                restored.shape[-1],
            ),
        )

        decoded = torch.cat(
            [decoded[:, :1, :], restored],
            dim=1,
        )
        decoded = (
            decoded
            + self.decoder_pos_embed.to(
                device=decoded.device,
                dtype=decoded.dtype,
            )
        )

        for block in self.decoder_blocks:
            decoded = block(decoded)

        decoded = self.decoder_norm(decoded)
        prediction = self.decoder_pred(decoded)
        return prediction[:, 1:, :]

    def forward_loss(
        self,
        images: Tensor,
        prediction: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """只在被遮挡 patch 上计算重建 MSE。"""
        target = self.patchify(images)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            variance = target.var(dim=-1, keepdim=True)
            target = (
                target - mean
            ) / torch.sqrt(variance + 1.0e-6)

        loss = (prediction - target) ** 2
        loss = loss.mean(dim=-1)

        masked_count = mask.sum()

        if masked_count <= 0:
            raise RuntimeError("当前 batch 没有被遮挡 patch。")

        return (loss * mask).sum() / masked_count

    def forward(
        self,
        images: Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        返回：
        - 标量重建损失；
        - patch 预测 [B,256,16*16*13]；
        - mask [B,256]。
        """
        encoded, mask, ids_restore = self.forward_encoder(
            images,
            mask_ratio=mask_ratio,
        )
        prediction = self.forward_decoder(
            encoded,
            ids_restore,
        )
        loss = self.forward_loss(
            images,
            prediction,
            mask,
        )
        return loss, prediction, mask

    def export_encoder_slr_state_dict(self) -> dict[str, Tensor]:
        """只导出编码器 ViT 的 SLR 参数。"""
        return extract_slr_state_dict(self.vit)

    def slr_statistics(self) -> dict[str, int]:
        """分别统计编码器和整个 MAE 的 SLR 参数。"""
        encoder = count_slr_parameters(self.vit)
        entire_model = count_slr_parameters(self)

        return {
            "encoder_slr_parameters": encoder["slr_parameters"],
            "all_slr_parameters": entire_model["slr_parameters"],
            "total_trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "total_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
            ),
        }
