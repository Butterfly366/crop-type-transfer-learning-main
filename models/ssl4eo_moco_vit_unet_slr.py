"""
SSL4EO-S12 MoCo ViT-S/16 + SLR + SMP UNet。

该模型用于后续两个阶段：

1. SLR 监督微调基线；
2. 加载自监督 MAE 阶段导出的 SLR 参数后进行监督微调。

原始 SSL4EO ViT 参数保持冻结。默认向每个 Transformer block 的：

- attn.qkv
- attn.proj
- mlp.fc1
- mlp.fc2

注入 Scaled Low-Rank Linear。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from models.peft_layers import count_parameters, freeze_module
from models.slr_layers import (
    count_slr_parameters,
    extract_slr_state_dict,
    inject_slr_into_vit,
    load_slr_state_dict,
)
from models.ssl4eo_moco_vit_unet import SSL4EOMoCoViTUNet


class SSL4EOMoCoViTUNetSLR(SSL4EOMoCoViTUNet):
    """
    SSL4EO-S12 MoCo ViT-S/16 + SLR + UNet。

    冻结：
    - SSL4EO ViT 原始参数。

    训练：
    - SLR in_scaler；
    - SLR out_scaler；
    - SLR down/up；
    - feature_norms；
    - ScaleAdapter；
    - UNet decoder；
    - segmentation head。
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 6,
        image_size: int = 256,
        slr_rank: int = 16,
        slr_blocks: Sequence[int] | None = None,
        slr_qkv: bool = True,
        slr_attn_proj: bool = True,
        slr_mlp_fc1: bool = True,
        slr_mlp_fc2: bool = True,
        slr_checkpoint_path: str | None = None,
        strict_slr_loading: bool = True,
    ) -> None:
        # 先创建原始 SSL4EO ViT-UNet，并加载预训练权重。
        super().__init__(
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            image_size=image_size,
            freeze_vit=True,
        )

        # 再次明确冻结完整 ViT。
        freeze_module(self.encoder.vit)

        self.slr_rank = int(slr_rank)
        self.slr_blocks = inject_slr_into_vit(
            vit=self.encoder.vit,
            rank=self.slr_rank,
            block_indices=slr_blocks,
            adapt_qkv=slr_qkv,
            adapt_attn_proj=slr_attn_proj,
            adapt_mlp_fc1=slr_mlp_fc1,
            adapt_mlp_fc2=slr_mlp_fc2,
        )

        self.slr_qkv = bool(slr_qkv)
        self.slr_attn_proj = bool(slr_attn_proj)
        self.slr_mlp_fc1 = bool(slr_mlp_fc1)
        self.slr_mlp_fc2 = bool(slr_mlp_fc2)

        self.peft_method = "slr"
        self.loaded_slr_checkpoint_path = None

        if slr_checkpoint_path is not None:
            self.load_slr_checkpoint(
                checkpoint_path=slr_checkpoint_path,
                strict=strict_slr_loading,
            )

        self.slr_parameter_statistics = count_slr_parameters(
            self.encoder.vit
        )
        self.parameter_statistics = count_parameters(self)

    def load_slr_checkpoint(
        self,
        checkpoint_path: str,
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        """
        加载 SLR-only checkpoint。

        支持三种文件结构：

        1. 直接保存 SLR state_dict；
        2. {"slr_state_dict": ...}；
        3. Lightning checkpoint 中的 {"state_dict": ...}，
           其中键包含 encoder.vit 或 model.encoder.vit 前缀。
        """
        path = Path(checkpoint_path).expanduser()

        if not path.is_file():
            raise FileNotFoundError(
                f"找不到 SLR checkpoint：{path}"
            )

        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

        state = self._extract_slr_state_from_checkpoint(
            checkpoint
        )

        missing, unexpected = load_slr_state_dict(
            self.encoder.vit,
            state,
            strict=strict,
        )

        self.loaded_slr_checkpoint_path = str(
            path.resolve()
        )

        return missing, unexpected

    def export_slr_state_dict(self) -> dict[str, Tensor]:
        """导出当前 ViT 中的 SLR-only 参数。"""
        return extract_slr_state_dict(
            self.encoder.vit
        )

    @staticmethod
    def _extract_slr_state_from_checkpoint(
        checkpoint: object,
    ) -> dict[str, Tensor]:
        """从不同 checkpoint 格式中提取 SLR 参数。"""
        if not isinstance(checkpoint, dict):
            raise TypeError(
                "SLR checkpoint 顶层必须是字典。"
            )

        if "slr_state_dict" in checkpoint:
            state = checkpoint["slr_state_dict"]

            if not isinstance(state, dict):
                raise TypeError(
                    "slr_state_dict 必须是字典。"
                )

            return state

        if "state_dict" in checkpoint:
            raw_state = checkpoint["state_dict"]

            if not isinstance(raw_state, dict):
                raise TypeError(
                    "Lightning state_dict 必须是字典。"
                )

            prefixes = (
                "model.encoder.vit.",
                "encoder.vit.",
                "model.vit.",
                "vit.",
            )

            result: dict[str, Tensor] = {}

            for key, value in raw_state.items():
                normalized_key = None

                for prefix in prefixes:
                    if key.startswith(prefix):
                        normalized_key = key[len(prefix):]
                        break

                if normalized_key is None:
                    continue

                if ".base_linear." in normalized_key:
                    continue

                if any(
                    token in normalized_key
                    for token in (
                        ".in_scaler",
                        ".out_scaler",
                        ".down.",
                        ".up.",
                    )
                ):
                    result[normalized_key] = value

            if not result:
                raise RuntimeError(
                    "Lightning checkpoint 中未找到 SLR 参数。"
                )

            return result

        # 直接保存的 SLR state_dict。
        if checkpoint and all(
            isinstance(key, str)
            for key in checkpoint
        ):
            return checkpoint

        raise RuntimeError(
            "无法识别 SLR checkpoint 格式。"
        )

    def forward(self, x: Tensor) -> Tensor:
        """执行 SLR ViT 编码和 UNet 分割。"""
        return super().forward(x)
