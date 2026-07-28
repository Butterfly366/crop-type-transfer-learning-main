"""
SSL4EO-S12 MoCo ViT-S/16 + LoRA-REIN + SMP UNet。

REIN 插入位置：
    每个 Transformer block 输出之后。

执行顺序：
    tokens = block(tokens)
    tokens = reins(tokens, layer=block_index)

然后在 blocks [1, 4, 7, 9, 11] 提取经过 REIN 调整后的中间特征，
继续使用项目现有的 feature_norms、ScaleAdapter、UNet decoder 和
segmentation head。
"""

from __future__ import annotations

from collections.abc import Sequence

import segmentation_models_pytorch as smp
from torch import Tensor, nn

from models.peft_layers import count_parameters
from models.rein_layers import LoRAReins, Reins, count_rein_parameters
from models.ssl4eo_moco_vit_unet import SSL4EOMoCoViTS16Encoder


class SSL4EOMoCoViTS16REINEncoder(SSL4EOMoCoViTS16Encoder):
    """在冻结 SSL4EO ViT 的每个 block 后加入 REIN。"""

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        selected_blocks: Sequence[int] = (1, 4, 7, 9, 11),
        rein_type: str = "lora_rein",
        token_length: int = 100,
        lora_dim: int = 16,
        use_softmax: bool = True,
        scale_init: float = 0.001,
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            selected_blocks=selected_blocks,
            freeze_vit=True,
        )

        normalized_type = rein_type.strip().lower()

        common_kwargs = {
            "num_layers": len(self.vit.blocks),
            "embed_dims": self.embed_dim,
            "patch_size": self.patch_size,
            "token_length": int(token_length),
            "use_softmax": bool(use_softmax),
            "scale_init": float(scale_init),
        }

        if normalized_type in {
            "lora_rein",
            "lorarein",
            "lora-rein",
        }:
            self.reins = LoRAReins(
                lora_dim=int(lora_dim),
                **common_kwargs,
            )
            self.rein_type = "lora_rein"

        elif normalized_type in {
            "rein",
            "full_rein",
            "full-rein",
        }:
            self.reins = Reins(**common_kwargs)
            self.rein_type = "rein"

        else:
            raise ValueError(
                "rein_type 必须为 'lora_rein' 或 'rein'，"
                f"当前为 {rein_type!r}。"
            )

        self.token_length = int(token_length)
        self.lora_dim = int(lora_dim)
        self.use_softmax = bool(use_softmax)
        self.scale_init = float(scale_init)

        # super().__init__ 已冻结完整 ViT。
        # REIN、feature_norms 和 ScaleAdapter 保持可训练。
        self.rein_parameter_statistics = count_rein_parameters(
            self.reins
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.ndim != 4:
            raise ValueError(
                f"输入应为 [B,C,H,W]，实际为 {tuple(x.shape)}。"
            )

        if x.shape[1] != 13:
            raise ValueError(
                f"输入应有 13 个波段，实际为 {x.shape[1]}。"
            )

        if x.shape[-2:] != (
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                "当前编码器固定处理 "
                f"{self.image_size}×{self.image_size} 输入，"
                f"实际为 {tuple(x.shape[-2:])}。"
            )

        output_features: list[Tensor] = [x]
        tokens = self._prepare_tokens(x)

        selected_tokens: list[Tensor] = []
        selected_set = set(self.selected_blocks)

        for block_index, block in enumerate(
            self.vit.blocks
        ):
            tokens = block(tokens)

            # timm ViT token 为 [B,N,C]，包含 CLS token。
            tokens = self.reins(
                tokens,
                layer=block_index,
                batch_first=True,
                has_cls_token=True,
            )

            if block_index in selected_set:
                selected_tokens.append(tokens)

        if len(selected_tokens) != 5:
            raise RuntimeError(
                "中间特征抽取数量不正确："
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


class SSL4EOMoCoViTUNetREIN(nn.Module):
    """SSL4EO-S12 MoCo ViT-S/16 + LoRA-REIN + UNet。"""

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 6,
        image_size: int = 256,
        rein_type: str = "lora_rein",
        token_length: int = 100,
        lora_dim: int = 16,
        use_softmax: bool = True,
        scale_init: float = 0.001,
        selected_blocks: Sequence[int] = (1, 4, 7, 9, 11),
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

        self.encoder = SSL4EOMoCoViTS16REINEncoder(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            selected_blocks=selected_blocks,
            rein_type=rein_type,
            token_length=token_length,
            lora_dim=lora_dim,
            use_softmax=use_softmax,
            scale_init=scale_init,
        )

        self.decoder = base_unet.decoder
        self.segmentation_head = (
            base_unet.segmentation_head
        )

        self.rein_type = self.encoder.rein_type
        self.token_length = int(token_length)
        self.lora_dim = int(lora_dim)
        self.use_softmax = bool(use_softmax)
        self.scale_init = float(scale_init)
        self.selected_blocks = tuple(
            int(index)
            for index in selected_blocks
        )

        self.parameter_statistics = count_parameters(
            self
        )
        self.rein_parameter_statistics = (
            self.encoder.rein_parameter_statistics
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        return self.segmentation_head(
            decoder_output
        )
