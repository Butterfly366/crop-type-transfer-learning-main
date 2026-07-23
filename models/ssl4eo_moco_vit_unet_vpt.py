from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from models.peft_layers import count_parameters, freeze_module
from models.ssl4eo_moco_vit_unet import (
    SSL4EOMoCoViTS16Encoder,
    SSL4EOMoCoViTUNet,
)


class SSL4EOMoCoViTS16VPTEncoder(SSL4EOMoCoViTS16Encoder):
    """SSL4EO-S12 MoCo ViT-S/16 encoder with shallow/deep VPT."""

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        selected_blocks: Sequence[int] = (1, 4, 7, 9, 11),
        prompt_length: int = 10,
        vpt_type: str = "deep",
        prompt_dropout: float = 0.0,
    ) -> None:
        if prompt_length <= 0:
            raise ValueError(f"prompt_length 必须大于 0，当前为 {prompt_length}。")

        vpt_type = str(vpt_type).strip().lower()
        if vpt_type not in {"deep", "shallow"}:
            raise ValueError("vpt_type 只能是 'deep' 或 'shallow'。")

        if not 0.0 <= prompt_dropout < 1.0:
            raise ValueError("prompt_dropout 必须位于 [0, 1)。")

        super().__init__(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            selected_blocks=selected_blocks,
            freeze_vit=True,
        )
        freeze_module(self.vit)

        self.prompt_length = int(prompt_length)
        self.vpt_type = vpt_type
        self.prompt_dropout = nn.Dropout(float(prompt_dropout))

        num_layers = len(self.vit.blocks)
        prompt_shape = (
            (num_layers, self.prompt_length, self.embed_dim)
            if self.vpt_type == "deep"
            else (1, self.prompt_length, self.embed_dim)
        )
        self.prompt_embeddings = nn.Parameter(torch.empty(prompt_shape))
        self._reset_prompt_parameters()

    def _reset_prompt_parameters(self) -> None:
        patch_area = self.patch_size * self.patch_size
        bound = math.sqrt(6.0 / float(3 * patch_area + self.embed_dim))
        nn.init.uniform_(self.prompt_embeddings, -bound, bound)

    def _expand_prompt(
        self,
        block_index: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        prompt_index = block_index if self.vpt_type == "deep" else 0
        prompt = self.prompt_dropout(self.prompt_embeddings[prompt_index])
        prompt = prompt.to(device=device, dtype=dtype)
        return prompt.unsqueeze(0).expand(batch_size, -1, -1)

    def _insert_prompt(self, base_tokens: Tensor, block_index: int) -> Tensor:
        prompt = self._expand_prompt(
            block_index=block_index,
            batch_size=base_tokens.shape[0],
            device=base_tokens.device,
            dtype=base_tokens.dtype,
        )
        return torch.cat(
            [base_tokens[:, :1, :], prompt, base_tokens[:, 1:, :]],
            dim=1,
        )

    def _remove_prompt(self, prompted_tokens: Tensor) -> Tensor:
        if prompted_tokens.shape[1] <= 1 + self.prompt_length:
            raise RuntimeError("VPT token 数量异常。")
        return torch.cat(
            [
                prompted_tokens[:, :1, :],
                prompted_tokens[:, 1 + self.prompt_length :, :],
            ],
            dim=1,
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        if x.ndim != 4:
            raise ValueError(f"输入应为 [B,C,H,W]，实际为 {tuple(x.shape)}。")
        if x.shape[1] != 13:
            raise ValueError(f"输入应有 13 个波段，实际为 {x.shape[1]}。")
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"输入应为 {self.image_size}×{self.image_size}，"
                f"实际为 {tuple(x.shape[-2:])}。"
            )

        output_features: list[Tensor] = [x]
        base_tokens = self._prepare_tokens(x)
        selected_tokens: list[Tensor] = []
        selected_set = set(self.selected_blocks)

        if self.vpt_type == "shallow":
            prompted_tokens = self._insert_prompt(base_tokens, 0)
            for block_index, block in enumerate(self.vit.blocks):
                prompted_tokens = block(prompted_tokens)
                if block_index in selected_set:
                    selected_tokens.append(self._remove_prompt(prompted_tokens))
        else:
            current_base_tokens = base_tokens
            for block_index, block in enumerate(self.vit.blocks):
                prompted_tokens = self._insert_prompt(
                    current_base_tokens,
                    block_index,
                )
                prompted_tokens = block(prompted_tokens)
                current_base_tokens = self._remove_prompt(prompted_tokens)
                if block_index in selected_set:
                    selected_tokens.append(current_base_tokens)

        if len(selected_tokens) != 5:
            raise RuntimeError(
                f"中间特征数量错误：期望 5，实际 {len(selected_tokens)}。"
            )

        for tokens_i, norm_i, adapter_i in zip(
            selected_tokens,
            self.feature_norms,
            self.adapters,
        ):
            tokens_i = norm_i(tokens_i)
            feature_map = self._tokens_to_feature_map(tokens_i)
            output_features.append(adapter_i(feature_map))

        return output_features


class SSL4EOMoCoViTUNetVPT(SSL4EOMoCoViTUNet):
    """SSL4EO-S12 MoCo ViT-S/16 + VPT + SMP UNet."""

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 6,
        image_size: int = 256,
        prompt_length: int = 10,
        vpt_type: str = "deep",
        prompt_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            image_size=image_size,
            freeze_vit=True,
        )

        self.encoder = SSL4EOMoCoViTS16VPTEncoder(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            selected_blocks=(1, 4, 7, 9, 11),
            prompt_length=prompt_length,
            vpt_type=vpt_type,
            prompt_dropout=prompt_dropout,
        )

        self.peft_method = f"vpt_{self.encoder.vpt_type}"
        self.prompt_length = int(prompt_length)
        self.vpt_type = self.encoder.vpt_type
        self.prompt_dropout = float(prompt_dropout)
        self.parameter_statistics = count_parameters(self)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
