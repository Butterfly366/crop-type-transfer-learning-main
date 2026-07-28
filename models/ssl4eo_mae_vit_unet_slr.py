"""
SSL4EO-S12 MAE ViT-S/16 + GDA-SLR + SMP UNet。

监督阶段流程
------------
1. 加载 SSL4EO-S12 13 波段 MAE ViT-S/16 完整预训练 checkpoint；
2. 只保留 MAE encoder，MAE decoder 不参与语义分割；
3. 将 encoder 位置编码从 224 输入对应的 14×14 网格，
   bicubic 插值到 256 输入对应的 16×16 网格；
4. 冻结原始 MAE encoder 参数；
5. 向 encoder 的 qkv、attn.proj、mlp.fc1、mlp.fc2 注入 SLR；
6. 向 patch embedding 注入 GDA 风格卷积 SLR；
7. 加载自监督阶段 checkpoint 中的 encoder SLR 参数；
8. LayerNorm、cls_token、SLR、特征适配器、UNet decoder、
   segmentation head 参与监督训练。

注意
----
第一阶段 checkpoint 同时包含 encoder 和 MAE decoder 的 SLR 参数。
监督阶段只加载：
- patch_embed.proj.*
- blocks.*
- cls_token（若 checkpoint 主 state_dict 中包含）
- encoder LayerNorm（通过监督 checkpoint 正常保存，不从 slr_state_dict 加载）

decoder_embed、decoder_blocks、decoder_pred 等 MAE decoder SLR 参数会被忽略。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
from timm.models.vision_transformer import Block, PatchEmbed
from torch import Tensor, nn
from torch.nn import functional as F

from models.peft_layers import count_parameters
from models.slr_layers import (
    ScaledLowRankLinear,
    count_slr_parameters,
    inject_slr_into_vit,
)
from models.ssl4eo_mae_vit_slr import (
    ScaledLowRankPatchEmbed,
    unfreeze_layer_norms,
)
from models.ssl4eo_moco_vit_unet import ScaleAdapter


class SSL4EOMAEViTS16SLREncoder(nn.Module):
    """
    兼容 SMP UNet encoder 接口的 SSL4EO MAE ViT-S/16 + SLR。

    输出特征：
        feature[0] = [B,   13, 256, 256]
        feature[1] = [B,   64, 128, 128]
        feature[2] = [B,  256,  64,  64]
        feature[3] = [B,  512,  32,  32]
        feature[4] = [B, 1024,  16,  16]
        feature[5] = [B, 2048,   8,   8]
    """

    out_channels = [13, 64, 256, 512, 1024, 2048]
    output_stride = 32

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
    ) -> None:
        super().__init__()

        if image_size % 16 != 0:
            raise ValueError("image_size 必须能够被 16 整除。")
        if slr_rank <= 0:
            raise ValueError("slr_rank 必须大于 0。")

        self.image_size = int(image_size)
        self.patch_size = 16
        self.embed_dim = 384
        self._in_channels = 13
        self._depth = 5

        self.selected_blocks = tuple(
            int(index)
            for index in selected_blocks
        )
        self._validate_selected_blocks()

        self.slr_rank = int(slr_rank)
        self.patch_embed_adapter = bool(patch_embed_adapter)
        self.norm_trainable = bool(norm_trainable)
        self.train_cls_token = bool(train_cls_token)

        num_patches = (
            self.image_size // self.patch_size
        ) ** 2

        # 建立与 SSL4EO 官方 MAE ViT-S/16 encoder 匹配的结构。
        self.patch_embed = PatchEmbed(
            img_size=self.image_size,
            patch_size=self.patch_size,
            in_chans=13,
            embed_dim=self.embed_dim,
        )
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.embed_dim)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                self.embed_dim,
            ),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=6,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=lambda dim: nn.LayerNorm(
                        dim,
                        eps=1.0e-6,
                    ),
                )
                for _ in range(12)
            ]
        )
        self.norm = nn.LayerNorm(
            self.embed_dim,
            eps=1.0e-6,
        )

        self._load_ssl4eo_mae_encoder_checkpoint(
            checkpoint_path
        )

        # 冻结原始 MAE encoder。
        for parameter in self.parameters():
            parameter.requires_grad = False

        # encoder Transformer 中注入线性 SLR。
        self.slr_blocks = inject_slr_into_vit(
            vit=self,
            rank=self.slr_rank,
            block_indices=slr_blocks,
            adapt_qkv=True,
            adapt_attn_proj=True,
            adapt_mlp_fc1=True,
            adapt_mlp_fc2=True,
        )

        # patch embedding 注入 GDA 卷积型 SLR。
        if self.patch_embed_adapter:
            self.patch_embed.proj = ScaledLowRankPatchEmbed(
                self.patch_embed.proj,
                rank=self.slr_rank,
            )

        if self.norm_trainable:
            unfreeze_layer_norms(self)

        self.cls_token.requires_grad = self.train_cls_token
        self.pos_embed.requires_grad = False

        # 中间层特征归一化和多尺度适配器属于分割任务新增模块，
        # 默认全部可训练。
        self.feature_norms = nn.ModuleList(
            [
                nn.LayerNorm(
                    self.embed_dim,
                    eps=1.0e-6,
                )
                for _ in self.selected_blocks
            ]
        )
        self.adapters = nn.ModuleList(
            [
                ScaleAdapter(
                    in_channels=384,
                    out_channels=64,
                    target_size=(128, 128),
                ),
                ScaleAdapter(
                    in_channels=384,
                    out_channels=256,
                    target_size=(64, 64),
                ),
                ScaleAdapter(
                    in_channels=384,
                    out_channels=512,
                    target_size=(32, 32),
                ),
                ScaleAdapter(
                    in_channels=384,
                    out_channels=1024,
                    target_size=(16, 16),
                ),
                ScaleAdapter(
                    in_channels=384,
                    out_channels=2048,
                    target_size=(8, 8),
                ),
            ]
        )

        self.loaded_slr_checkpoint_path: str | None = None
        self.slr_load_missing: list[str] = []
        self.slr_load_unexpected: list[str] = []

        if slr_checkpoint_path is not None:
            missing, unexpected = self.load_slr_checkpoint(
                checkpoint_path=slr_checkpoint_path,
                strict=strict_slr_loading,
            )
            self.slr_load_missing = missing
            self.slr_load_unexpected = unexpected

        self.slr_parameter_statistics = (
            self._count_encoder_slr_parameters()
        )

    def _validate_selected_blocks(self) -> None:
        """检查用于 UNet 多尺度特征的 block 编号。"""

        if len(self.selected_blocks) != 5:
            raise ValueError(
                "selected_blocks 必须包含 5 个 block 编号。"
            )
        if tuple(sorted(self.selected_blocks)) != self.selected_blocks:
            raise ValueError(
                "selected_blocks 必须按从浅到深排序。"
            )
        if len(set(self.selected_blocks)) != len(self.selected_blocks):
            raise ValueError(
                "selected_blocks 不能包含重复编号。"
            )

        for block_index in self.selected_blocks:
            if block_index < 0 or block_index >= 12:
                raise ValueError(
                    f"非法 block 编号 {block_index}，"
                    "有效范围为 0～11。"
                )

    @staticmethod
    def _extract_model_state_dict(
        checkpoint: object,
    ) -> dict[str, Tensor]:
        """从 SSL4EO MAE checkpoint 中提取完整模型 state_dict。"""

        if not isinstance(checkpoint, dict):
            raise TypeError("MAE checkpoint 顶层必须是字典。")

        if "model" in checkpoint:
            state = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint

        if not isinstance(state, dict):
            raise TypeError("MAE state_dict 必须是字典。")

        normalized: dict[str, Tensor] = {}

        for key, value in state.items():
            if not isinstance(key, str) or not isinstance(value, Tensor):
                continue

            new_key = key

            for prefix in ("module.", "model."):
                while new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]

            normalized[new_key] = value

        if not normalized:
            raise RuntimeError(
                "没有从 MAE checkpoint 提取到模型参数。"
            )

        return normalized

    @staticmethod
    def _interpolate_position_embedding(
        source_pos_embed: Tensor,
        target_pos_embed: Tensor,
    ) -> Tensor:
        """
        保留 CLS token，只对 patch 网格做 bicubic 插值。

        与项目现有 MoCo encoder 的位置编码插值逻辑一致。
        """

        if source_pos_embed.shape == target_pos_embed.shape:
            return source_pos_embed

        if source_pos_embed.ndim != 3:
            raise ValueError("checkpoint pos_embed 必须为三维张量。")
        if target_pos_embed.ndim != 3:
            raise ValueError("目标 pos_embed 必须为三维张量。")
        if source_pos_embed.shape[-1] != target_pos_embed.shape[-1]:
            raise ValueError(
                "位置编码嵌入维度不匹配："
                f"{source_pos_embed.shape[-1]} vs "
                f"{target_pos_embed.shape[-1]}。"
            )

        num_prefix_tokens = 1
        embedding_dim = source_pos_embed.shape[-1]

        cls_position = source_pos_embed[
            :,
            :num_prefix_tokens,
            :,
        ]
        patch_positions = source_pos_embed[
            :,
            num_prefix_tokens:,
            :,
        ]

        old_num_patches = patch_positions.shape[1]
        new_num_patches = (
            target_pos_embed.shape[1]
            - num_prefix_tokens
        )

        old_grid_size = int(old_num_patches**0.5)
        new_grid_size = int(new_num_patches**0.5)

        if old_grid_size**2 != old_num_patches:
            raise ValueError(
                "checkpoint patch 位置编码不能恢复为正方形网格。"
            )
        if new_grid_size**2 != new_num_patches:
            raise ValueError(
                "目标 patch 位置编码不能恢复为正方形网格。"
            )

        patch_positions = patch_positions.reshape(
            1,
            old_grid_size,
            old_grid_size,
            embedding_dim,
        )
        patch_positions = patch_positions.permute(
            0,
            3,
            1,
            2,
        )
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

    def _load_ssl4eo_mae_encoder_checkpoint(
        self,
        checkpoint_path: str,
    ) -> None:
        """加载完整 MAE checkpoint 中的 encoder 参数。"""

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
        full_state = self._extract_model_state_dict(
            checkpoint
        )

        encoder_prefixes = (
            "patch_embed.",
            "cls_token",
            "pos_embed",
            "blocks.",
            "norm.",
        )

        encoder_state = {
            key: value
            for key, value in full_state.items()
            if key.startswith(encoder_prefixes)
        }

        required_top_level = {
            "cls_token",
            "pos_embed",
            "patch_embed.proj.weight",
            "patch_embed.proj.bias",
            "norm.weight",
            "norm.bias",
        }

        missing_required = sorted(
            required_top_level - set(encoder_state)
        )
        if missing_required:
            raise RuntimeError(
                "MAE checkpoint 缺少关键 encoder 参数：\n"
                + "\n".join(missing_required)
            )

        encoder_state["pos_embed"] = (
            self._interpolate_position_embedding(
                source_pos_embed=encoder_state["pos_embed"],
                target_pos_embed=self.pos_embed,
            )
        )

        load_result = self.load_state_dict(
            encoder_state,
            strict=False,
        )

        # 此时 feature_norms/adapters 尚未创建，因此不应产生它们的缺失项。
        if load_result.missing_keys:
            raise RuntimeError(
                "MAE encoder 权重存在缺失参数：\n"
                + "\n".join(load_result.missing_keys)
            )
        if load_result.unexpected_keys:
            raise RuntimeError(
                "MAE encoder 权重存在多余参数：\n"
                + "\n".join(load_result.unexpected_keys)
            )

    @staticmethod
    def _extract_slr_state_from_checkpoint(
        checkpoint: object,
    ) -> dict[str, Tensor]:
        """从第一阶段 checkpoint 中提取 SLR-only state_dict。"""

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "SLR checkpoint 顶层必须是字典。"
            )

        if "slr_state_dict" in checkpoint:
            state = checkpoint["slr_state_dict"]
        elif "state_dict" in checkpoint:
            raw_state = checkpoint["state_dict"]

            if not isinstance(raw_state, dict):
                raise TypeError(
                    "Lightning state_dict 必须是字典。"
                )

            prefixes = (
                "model.",
                "encoder.",
                "model.encoder.",
            )
            state = {}

            for key, value in raw_state.items():
                if not isinstance(key, str) or not isinstance(value, Tensor):
                    continue

                normalized = key

                for prefix in prefixes:
                    if normalized.startswith(prefix):
                        normalized = normalized[len(prefix):]
                        break

                if any(
                    token in normalized
                    for token in (
                        ".in_scaler",
                        ".out_scaler",
                        ".down.",
                        ".up.",
                    )
                ):
                    if (
                        ".base_linear." not in normalized
                        and ".base_conv." not in normalized
                    ):
                        state[normalized] = value
        else:
            state = checkpoint

        if not isinstance(state, dict):
            raise TypeError("SLR state_dict 必须是字典。")

        result = {
            str(key): value
            for key, value in state.items()
            if isinstance(key, str)
            and isinstance(value, Tensor)
        }

        if not result:
            raise RuntimeError(
                "checkpoint 中没有可用的 SLR 参数。"
            )

        return result

    def _expected_encoder_slr_keys(self) -> set[str]:
        """返回监督 encoder 当前期望的全部 SLR 参数键。"""

        expected: set[str] = set()

        for module_name, module in self.named_modules():
            if isinstance(module, ScaledLowRankLinear):
                for parameter_name, _ in module.named_parameters():
                    if parameter_name.startswith("base_linear."):
                        continue
                    expected.add(
                        f"{module_name}.{parameter_name}"
                    )

            elif isinstance(module, ScaledLowRankPatchEmbed):
                for parameter_name, _ in module.named_parameters():
                    if parameter_name.startswith("base_conv."):
                        continue
                    expected.add(
                        f"{module_name}.{parameter_name}"
                    )

        return expected

    @staticmethod
    def _is_encoder_slr_key(key: str) -> bool:
        """判断第一阶段 SLR 键是否属于 encoder。"""

        return (
            key.startswith("blocks.")
            or key.startswith("patch_embed.proj.")
        )

    def load_slr_checkpoint(
        self,
        checkpoint_path: str,
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        """
        加载第一阶段自监督 checkpoint 中的 encoder SLR。

        MAE decoder 的 SLR 参数会被过滤掉，不视为 unexpected。
        """

        path = Path(checkpoint_path).expanduser()

        if not path.is_file():
            raise FileNotFoundError(
                f"找不到 SLR checkpoint：{path}"
            )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
        full_slr_state = (
            self._extract_slr_state_from_checkpoint(
                checkpoint
            )
        )

        encoder_slr_state = {
            key: value
            for key, value in full_slr_state.items()
            if self._is_encoder_slr_key(key)
        }

        if not encoder_slr_state:
            raise RuntimeError(
                "第一阶段 checkpoint 中没有找到 encoder SLR 参数。"
            )

        expected = self._expected_encoder_slr_keys()
        provided = set(encoder_slr_state)

        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)

        if strict and (missing or unexpected):
            messages: list[str] = []

            if missing:
                messages.append(
                    "缺失 encoder SLR 参数：\n"
                    + "\n".join(missing)
                )
            if unexpected:
                messages.append(
                    "多余 encoder SLR 参数：\n"
                    + "\n".join(unexpected)
                )

            raise RuntimeError("\n\n".join(messages))

        current_state = self.state_dict()

        for key in expected & provided:
            source = encoder_slr_state[key]
            target = current_state[key]

            if source.shape != target.shape:
                raise RuntimeError(
                    f"SLR 参数形状不匹配：{key}，"
                    f"checkpoint={tuple(source.shape)}，"
                    f"model={tuple(target.shape)}。"
                )

            current_state[key] = source

        self.load_state_dict(
            current_state,
            strict=True,
        )

        self.loaded_slr_checkpoint_path = str(
            path.resolve()
        )

        return missing, unexpected

    def _count_encoder_slr_parameters(
        self,
    ) -> dict[str, int]:
        """统计 encoder 的线性和 patch embedding SLR 参数。"""

        linear_stats = count_slr_parameters(self)

        patch_slr = 0
        if isinstance(
            self.patch_embed.proj,
            ScaledLowRankPatchEmbed,
        ):
            patch_slr = sum(
                parameter.numel()
                for name, parameter
                in self.patch_embed.proj.named_parameters()
                if parameter.requires_grad
                and not name.startswith("base_conv.")
            )

        return {
            "linear_slr_parameters": linear_stats[
                "slr_parameters"
            ],
            "patch_embed_slr_parameters": patch_slr,
            "all_slr_parameters": (
                linear_stats["slr_parameters"]
                + patch_slr
            ),
            "encoder_trainable_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "encoder_total_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
            ),
        }

    def _prepare_tokens(self, x: Tensor) -> Tensor:
        """执行 patch embedding、CLS token 和位置编码。"""

        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed[:, 1:, :]

        cls_token = (
            self.cls_token
            + self.pos_embed[:, :1, :]
        )
        cls_tokens = cls_token.expand(
            x.shape[0],
            -1,
            -1,
        )

        return torch.cat(
            [cls_tokens, tokens],
            dim=1,
        )

    def _tokens_to_feature_map(
        self,
        tokens: Tensor,
    ) -> Tensor:
        """删除 CLS token 并恢复为 16×16 空间特征。"""

        if tokens.ndim != 3:
            raise ValueError(
                "tokens 必须是 [B,N,C] 三维张量。"
            )

        patch_tokens = tokens[:, 1:, :]

        grid_size = self.image_size // self.patch_size
        expected_patches = grid_size**2

        if patch_tokens.shape[1] != expected_patches:
            raise RuntimeError(
                "patch token 数量错误："
                f"期望 {expected_patches}，"
                f"实际 {patch_tokens.shape[1]}。"
            )

        feature_map = patch_tokens.reshape(
            patch_tokens.shape[0],
            grid_size,
            grid_size,
            self.embed_dim,
        )

        return feature_map.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

    def forward(self, x: Tensor) -> list[Tensor]:
        """返回 SMP UNet 所需的六级特征。"""

        if x.ndim != 4:
            raise ValueError(
                f"输入应为 [B,C,H,W]，实际为 {tuple(x.shape)}。"
            )
        if x.shape[1] != 13:
            raise ValueError(
                f"输入应为 13 波段，实际为 {x.shape[1]}。"
            )
        if x.shape[-2:] != (
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                f"输入尺寸必须为 {self.image_size}×"
                f"{self.image_size}，实际为 "
                f"{tuple(x.shape[-2:])}。"
            )

        output_features: list[Tensor] = [x]
        tokens = self._prepare_tokens(x)

        selected_tokens: list[Tensor] = []
        selected_set = set(self.selected_blocks)

        for block_index, block in enumerate(self.blocks):
            tokens = block(tokens)

            if block_index in selected_set:
                selected_tokens.append(tokens)

        if len(selected_tokens) != 5:
            raise RuntimeError(
                "中间特征数量错误："
                f"期望 5，实际 {len(selected_tokens)}。"
            )

        for tokens_i, norm_i, adapter_i in zip(
            selected_tokens,
            self.feature_norms,
            self.adapters,
        ):
            tokens_i = norm_i(tokens_i)
            feature_map = self._tokens_to_feature_map(
                tokens_i
            )
            output_features.append(
                adapter_i(feature_map)
            )

        return output_features


class SSL4EOMAEViTUNetSLR(nn.Module):
    """SSL4EO MAE ViT-S/16 + encoder SLR + SMP UNet。"""

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 6,
        image_size: int = 256,
        selected_blocks: Sequence[int] = (1, 4, 7, 9, 11),
        slr_rank: int = 16,
        slr_blocks: Sequence[int] | None = None,
        patch_embed_adapter: bool = True,
        norm_trainable: bool = True,
        train_cls_token: bool = True,
        slr_checkpoint_path: str | None = None,
        strict_slr_loading: bool = True,
    ) -> None:
        super().__init__()

        base_unet = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=13,
            classes=num_classes,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
            activation=None,
        )

        self.encoder = SSL4EOMAEViTS16SLREncoder(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            selected_blocks=selected_blocks,
            slr_rank=slr_rank,
            slr_blocks=slr_blocks,
            patch_embed_adapter=patch_embed_adapter,
            norm_trainable=norm_trainable,
            train_cls_token=train_cls_token,
            slr_checkpoint_path=slr_checkpoint_path,
            strict_slr_loading=strict_slr_loading,
        )

        self.decoder = base_unet.decoder
        self.segmentation_head = (
            base_unet.segmentation_head
        )

        self.peft_method = "slr"
        self.loaded_slr_checkpoint_path = (
            self.encoder.loaded_slr_checkpoint_path
        )
        self.slr_parameter_statistics = (
            self.encoder.slr_parameter_statistics
        )
        self.parameter_statistics = count_parameters(self)

    def forward(self, x: Tensor) -> Tensor:
        """执行 MAE ViT 编码、UNet 解码和分割预测。"""

        features = self.encoder(x)
        decoder_output = self.decoder(features)
        return self.segmentation_head(decoder_output)