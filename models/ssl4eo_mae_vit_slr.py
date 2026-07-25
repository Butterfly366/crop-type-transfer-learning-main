"""
SSL4EO-S12 MAE ViT-S/16 + GDA SLR 自监督域适配模型。

设计原则
--------
1. 加载 SSL4EO-S12 官方 13 波段 MAE ViT-S/16 完整 checkpoint；
2. 同时复用已经预训练的 encoder 和 MAE decoder；
3. 冻结所有原始预训练参数；
4. 按 GDA 官方配置向 encoder/decoder 线性层注入 SLR；
5. 向 patch embedding 注入卷积型 SLR；
6. LayerNorm、cls_token 和 mask_token 保持可训练；
7. encoder 与 decoder 位置编码使用和现有 MoCo 代码相同的 bicubic 网格插值；
8. 默认在全部 patch 上计算重建损失，对齐 GDA 配置。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from timm.models.vision_transformer import Block, PatchEmbed
from torch import Tensor, nn
from torch.nn import functional as F

from models.slr_layers import (
    ScaledLowRankLinear,
    count_slr_parameters,
    extract_slr_state_dict,
    inject_slr_into_vit,
)


class ScaledLowRankPatchEmbed(nn.Module):
    """GDA 风格的 patch embedding 卷积型 SLR adapter。"""

    def __init__(self, base_conv: nn.Conv2d, rank: int = 16) -> None:
        super().__init__()

        if not isinstance(base_conv, nn.Conv2d):
            raise TypeError(
                "ScaledLowRankPatchEmbed 只能包装 nn.Conv2d，"
                f"实际类型为 {type(base_conv).__name__}。"
            )
        if rank <= 0:
            raise ValueError(f"SLR rank 必须大于 0，当前为 {rank}。")
        if base_conv.kernel_size != (16, 16):
            raise ValueError(
                "GDA patch embedding SLR 当前要求原始卷积 kernel_size=16。"
            )
        if base_conv.stride != (16, 16):
            raise ValueError(
                "GDA patch embedding SLR 当前要求原始卷积 stride=16。"
            )

        self.rank = int(rank)
        self.base_conv = base_conv

        for parameter in self.base_conv.parameters():
            parameter.requires_grad = False

        self.down = nn.Conv2d(
            in_channels=base_conv.in_channels,
            out_channels=self.rank,
            kernel_size=4,
            stride=4,
            bias=True,
        )
        self.up = nn.Conv2d(
            in_channels=self.rank,
            out_channels=base_conv.out_channels,
            kernel_size=4,
            stride=4,
            bias=True,
        )
        self.out_scaler = nn.Parameter(
            torch.ones(base_conv.out_channels)
        )

        nn.init.normal_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    @property
    def weight(self) -> Tensor:
        return self.base_conv.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base_conv.bias

    def forward(self, x: Tensor) -> Tensor:
        base_output = self.base_conv(x)
        low_rank_output = self.up(self.down(x))

        if base_output.shape != low_rank_output.shape:
            raise RuntimeError(
                "patch embedding 主分支与 SLR 分支形状不一致："
                f"{tuple(base_output.shape)} vs "
                f"{tuple(low_rank_output.shape)}。"
            )

        output = base_output + low_rank_output
        return output * self.out_scaler.view(1, -1, 1, 1)


def unfreeze_layer_norms(module: nn.Module) -> None:
    """使模块中的全部 LayerNorm 参数可训练。"""
    for child in module.modules():
        if isinstance(child, nn.LayerNorm):
            for parameter in child.parameters():
                parameter.requires_grad = True


class SSL4EOMAEViTSLR(nn.Module):
    """SSL4EO-S12 MAE ViT-S/16 + GDA SLR。"""

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        in_channels: int = 13,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        slr_rank: int = 16,
        slr_blocks: Sequence[int] | None = None,
        patch_embed_adapter: bool = True,
        norm_trainable: bool = True,
        train_cls_mask_tokens: bool = True,
        loss_on_all_patches: bool = True,
        norm_pix_loss: bool = False,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size 必须能够被 patch_size 整除。")
        if in_channels != 13:
            raise ValueError(
                "SSL4EO-S12 S2-L1C MAE ViT-S/16 要求 13 波段输入。"
            )
        if patch_size != 16:
            raise ValueError(
                "SSL4EO-S12 MAE ViT-S/16 的 patch_size 必须为 16。"
            )
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio 必须位于 (0, 1)。")
        if slr_rank <= 0:
            raise ValueError("slr_rank 必须大于 0。")
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
        self.patch_embed_adapter = bool(patch_embed_adapter)
        self.norm_trainable = bool(norm_trainable)
        self.train_cls_mask_tokens = bool(train_cls_mask_tokens)
        self.loss_on_all_patches = bool(loss_on_all_patches)
        self.norm_pix_loss = bool(norm_pix_loss)

        encoder_embed_dim = 384
        encoder_depth = 12
        encoder_num_heads = 6

        self.patch_embed = PatchEmbed(
            img_size=self.image_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=encoder_embed_dim,
        )
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, encoder_embed_dim)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches + 1,
                encoder_embed_dim,
            ),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=encoder_embed_dim,
                    num_heads=encoder_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=lambda dim: nn.LayerNorm(
                        dim,
                        eps=1.0e-6,
                    ),
                )
                for _ in range(encoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(
            encoder_embed_dim,
            eps=1.0e-6,
        )

        self.decoder_embed = nn.Linear(
            encoder_embed_dim,
            decoder_embed_dim,
            bias=True,
        )
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, decoder_embed_dim)
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches + 1,
                decoder_embed_dim,
            ),
            requires_grad=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=lambda dim: nn.LayerNorm(
                        dim,
                        eps=1.0e-6,
                    ),
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(
            decoder_embed_dim,
            eps=1.0e-6,
        )
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.patch_size**2 * self.in_channels,
            bias=True,
        )

        self._load_ssl4eo_mae_checkpoint(checkpoint_path)

        # 冻结完整 MAE encoder/decoder 的原始预训练参数。
        for parameter in self.parameters():
            parameter.requires_grad = False

        self.slr_blocks = inject_slr_into_vit(
            vit=self,
            rank=self.slr_rank,
            block_indices=slr_blocks,
            adapt_qkv=True,
            adapt_attn_proj=True,
            adapt_mlp_fc1=True,
            adapt_mlp_fc2=True,
        )

        decoder_wrapper = nn.Module()
        decoder_wrapper.blocks = self.decoder_blocks
        inject_slr_into_vit(
            vit=decoder_wrapper,
            rank=self.slr_rank,
            block_indices=None,
            adapt_qkv=True,
            adapt_attn_proj=True,
            adapt_mlp_fc1=True,
            adapt_mlp_fc2=True,
        )

        self.decoder_embed = ScaledLowRankLinear(
            self.decoder_embed,
            rank=self.slr_rank,
        )
        self.decoder_pred = ScaledLowRankLinear(
            self.decoder_pred,
            rank=self.slr_rank,
        )

        if self.patch_embed_adapter:
            self.patch_embed.proj = ScaledLowRankPatchEmbed(
                self.patch_embed.proj,
                rank=self.slr_rank,
            )

        if self.norm_trainable:
            unfreeze_layer_norms(self)

        self.cls_token.requires_grad = self.train_cls_mask_tokens
        self.mask_token.requires_grad = self.train_cls_mask_tokens

    @staticmethod
    def _extract_model_state_dict(
        checkpoint: object,
    ) -> dict[str, Tensor]:
        """提取 SSL4EO-S12 MAE 完整模型权重。"""
        if not isinstance(checkpoint, dict):
            raise TypeError("MAE checkpoint 顶层必须是字典。")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise TypeError("MAE 模型权重必须是字典。")

        normalized: dict[str, Tensor] = {}
        prefixes = ("module.", "model.")

        for key, value in state_dict.items():
            if not isinstance(key, str) or not isinstance(value, Tensor):
                continue

            new_key = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        changed = True

            normalized[new_key] = value

        if not normalized:
            raise RuntimeError("没有从 checkpoint 中提取到模型参数。")

        return normalized

    @staticmethod
    def _interpolate_position_embedding_tensor(
        source_pos_embed: Tensor,
        target_pos_embed: Tensor,
        name: str,
    ) -> Tensor:
        """保留 CLS token，仅对 patch 网格做 bicubic 插值。"""
        if source_pos_embed.shape == target_pos_embed.shape:
            return source_pos_embed

        if source_pos_embed.ndim != 3 or target_pos_embed.ndim != 3:
            raise ValueError(f"{name} 必须是三维张量。")
        if source_pos_embed.shape[0] != 1:
            raise ValueError(f"{name} 的 batch 维度必须为 1。")
        if source_pos_embed.shape[-1] != target_pos_embed.shape[-1]:
            raise ValueError(
                f"{name} 的嵌入维度不匹配："
                f"{source_pos_embed.shape[-1]} vs "
                f"{target_pos_embed.shape[-1]}。"
            )

        num_prefix_tokens = 1
        embedding_dim = source_pos_embed.shape[-1]

        cls_position = source_pos_embed[:, :num_prefix_tokens, :]
        patch_positions = source_pos_embed[:, num_prefix_tokens:, :]

        old_num_patches = patch_positions.shape[1]
        new_num_patches = (
            target_pos_embed.shape[1] - num_prefix_tokens
        )

        old_grid_size = int(old_num_patches**0.5)
        new_grid_size = int(new_num_patches**0.5)

        if old_grid_size**2 != old_num_patches:
            raise ValueError(
                f"{name} 的 checkpoint patch 数量不能恢复为正方形网格。"
            )
        if new_grid_size**2 != new_num_patches:
            raise ValueError(
                f"{name} 的目标 patch 数量不能恢复为正方形网格。"
            )

        patch_positions = patch_positions.reshape(
            1,
            old_grid_size,
            old_grid_size,
            embedding_dim,
        )
        patch_positions = patch_positions.permute(0, 3, 1, 2)
        patch_positions = F.interpolate(
            patch_positions,
            size=(new_grid_size, new_grid_size),
            mode="bicubic",
            align_corners=False,
        )
        patch_positions = patch_positions.permute(
            0,
            2,
            3,
            1,
        ).reshape(
            1,
            new_grid_size * new_grid_size,
            embedding_dim,
        )

        return torch.cat(
            [cls_position, patch_positions],
            dim=1,
        )

    def _interpolate_checkpoint_position_embeddings(
        self,
        state_dict: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """同时插值 encoder 和 decoder 位置编码。"""
        state_dict = dict(state_dict)

        for key, target in (
            ("pos_embed", self.pos_embed),
            ("decoder_pos_embed", self.decoder_pos_embed),
        ):
            if key not in state_dict:
                raise KeyError(
                    f"SSL4EO MAE checkpoint 中缺少 {key}。"
                )

            state_dict[key] = (
                self._interpolate_position_embedding_tensor(
                    source_pos_embed=state_dict[key],
                    target_pos_embed=target,
                    name=key,
                )
            )

        return state_dict

    def _load_ssl4eo_mae_checkpoint(
        self,
        checkpoint_path: str,
    ) -> None:
        """加载 SSL4EO-S12 MAE ViT-S/16 完整 checkpoint。"""
        path = Path(checkpoint_path).expanduser()

        if not path.is_file():
            raise FileNotFoundError(
                f"找不到 SSL4EO MAE checkpoint：{path}"
            )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = self._extract_model_state_dict(checkpoint)
        state_dict = self._interpolate_checkpoint_position_embeddings(
            state_dict
        )

        load_result = self.load_state_dict(
            state_dict,
            strict=False,
        )

        allowed_unexpected = {
            key
            for key in load_result.unexpected_keys
            if key.startswith("head.")
        }
        unexpected = sorted(
            set(load_result.unexpected_keys) - allowed_unexpected
        )

        if load_result.missing_keys:
            raise RuntimeError(
                "SSL4EO MAE 权重存在缺失参数：\n"
                + "\n".join(load_result.missing_keys)
            )
        if unexpected:
            raise RuntimeError(
                "SSL4EO MAE 权重存在无法识别的参数：\n"
                + "\n".join(unexpected)
            )

    def patchify(self, images: Tensor) -> Tensor:
        """将影像转换为 [B, L, patch_size²×C]。"""
        if images.ndim != 4:
            raise ValueError("images 必须是 [B,C,H,W] 四维张量。")
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"输入波段数应为 {self.in_channels}，"
                f"当前为 {images.shape[1]}。"
            )
        if images.shape[2:] != (
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                f"输入空间尺寸必须为 "
                f"{self.image_size}×{self.image_size}。"
            )

        p = self.patch_size
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

    @staticmethod
    def random_masking(
        tokens: Tensor,
        mask_ratio: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """逐样本随机遮挡 patch token。"""
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
        """执行 patch embedding、随机遮挡和 MAE encoder。"""
        ratio = (
            self.mask_ratio
            if mask_ratio is None
            else float(mask_ratio)
        )

        tokens = self.patch_embed(images)
        tokens = tokens + self.pos_embed[:, 1:, :]

        visible, mask, ids_restore = self.random_masking(
            tokens,
            ratio,
        )

        cls_token = (
            self.cls_token
            + self.pos_embed[:, :1, :]
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

        for block in self.blocks:
            encoded = block(encoded)

        encoded = self.norm(encoded)
        return encoded, mask, ids_restore

    def forward_decoder(
        self,
        encoded: Tensor,
        ids_restore: Tensor,
    ) -> Tensor:
        """恢复 mask token 并重建全部 patch。"""
        decoded = self.decoder_embed(encoded)

        mask_tokens = self.mask_token.repeat(
            decoded.shape[0],
            ids_restore.shape[1] + 1 - decoded.shape[1],
            1,
        )
        restored = torch.cat(
            [decoded[:, 1:, :], mask_tokens],
            dim=1,
        )
        restored = torch.gather(
            restored,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(
                -1,
                -1,
                decoded.shape[-1],
            ),
        )
        decoded = torch.cat(
            [decoded[:, :1, :], restored],
            dim=1,
        )
        decoded = decoded + self.decoder_pos_embed

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
        """计算重建损失。默认对齐 GDA，在全部 patch 上计算。"""
        target = self.patchify(images)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            variance = target.var(dim=-1, keepdim=True)
            target = (
                target - mean
            ) / torch.sqrt(variance + 1.0e-6)

        loss = (prediction - target) ** 2
        loss = loss.mean(dim=-1)

        if self.loss_on_all_patches:
            return loss.mean()

        masked_count = mask.sum()
        if masked_count <= 0:
            raise RuntimeError("当前 batch 没有被遮挡 patch。")

        return (loss * mask).sum() / masked_count

    def forward(
        self,
        images: Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
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

    def export_slr_state_dict(self) -> dict[str, Tensor]:
        """导出当前模型的全部 SLR 参数。"""
        state = extract_slr_state_dict(self)

        if isinstance(
            self.patch_embed.proj,
            ScaledLowRankPatchEmbed,
        ):
            prefix = "patch_embed.proj."
            for name, value in self.patch_embed.proj.state_dict().items():
                if name.startswith("base_conv."):
                    continue
                state[prefix + name] = value.detach().cpu().clone()

        return state

    def slr_statistics(self) -> dict[str, int]:
        """统计总参数、可训练参数和 SLR 参数。"""
        stats = count_slr_parameters(self)

        patch_slr_parameters = 0
        if isinstance(
            self.patch_embed.proj,
            ScaledLowRankPatchEmbed,
        ):
            patch_slr_parameters = sum(
                parameter.numel()
                for name, parameter
                in self.patch_embed.proj.named_parameters()
                if parameter.requires_grad
                and not name.startswith("base_conv.")
            )

        return {
            "total_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
            ),
            "total_trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "linear_slr_parameters": stats["slr_parameters"],
            "patch_embed_slr_parameters": patch_slr_parameters,
            "all_slr_parameters": (
                stats["slr_parameters"]
                + patch_slr_parameters
            ),
        }