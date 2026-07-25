from __future__ import annotations

import math
from functools import reduce
from operator import mul

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Reins(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        token_length: int = 100,
        use_softmax: bool = True,
        scale_init: float = 0.001,
    ) -> None:
        super().__init__()
        if num_layers <= 0 or embed_dims <= 0 or patch_size <= 0:
            raise ValueError("num_layers、embed_dims、patch_size 必须大于 0。")
        if token_length < 2:
            raise ValueError("token_length 至少为 2。")

        self.num_layers = int(num_layers)
        self.embed_dims = int(embed_dims)
        self.patch_size = int(patch_size)
        self.token_length = int(token_length)
        self.use_softmax = bool(use_softmax)
        self.scale_init = float(scale_init)
        self._create_model()

    def _create_model(self) -> None:
        self.learnable_tokens = nn.Parameter(
            torch.empty(self.num_layers, self.token_length, self.embed_dims)
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)

        value = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens, -value, value)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            return self.learnable_tokens
        if layer < 0 or layer >= self.num_layers:
            raise IndexError(
                f"非法 REIN layer={layer}，有效范围为 0～{self.num_layers - 1}。"
            )
        return self.learnable_tokens[layer]

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor) -> Tensor:
        attention = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attention = attention * (self.embed_dims ** -0.5)
            attention = F.softmax(attention, dim=-1)

        delta_feature = torch.einsum(
            "nbm,mc->nbc",
            attention[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )
        return self.mlp_delta_f(delta_feature + feats)

    def forward(
        self,
        feats: Tensor,
        layer: int,
        batch_first: bool = False,
        has_cls_token: bool = True,
    ) -> Tensor:
        if feats.ndim != 3:
            raise ValueError("REIN 输入必须是三维 token 张量。")

        if batch_first:
            feats = feats.permute(1, 0, 2)

        if has_cls_token:
            cls_token, patch_tokens = torch.tensor_split(feats, [1], dim=0)
        else:
            cls_token = None
            patch_tokens = feats

        tokens = self.get_tokens(layer)
        patch_tokens = patch_tokens + self.forward_delta_feat(
            patch_tokens, tokens
        ) * self.scale

        if cls_token is not None:
            feats = torch.cat([cls_token, patch_tokens], dim=0)
        else:
            feats = patch_tokens

        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats


class LoRAReins(Reins):
    def __init__(self, lora_dim: int = 16, **kwargs) -> None:
        if lora_dim <= 0:
            raise ValueError("lora_dim 必须大于 0。")
        self.lora_dim = int(lora_dim)
        super().__init__(**kwargs)

    def _create_model(self) -> None:
        super()._create_model()
        del self.learnable_tokens

        self.learnable_tokens_a = nn.Parameter(
            torch.empty(self.num_layers, self.token_length, self.lora_dim)
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty(self.num_layers, self.lora_dim, self.embed_dims)
        )

        value = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a, -value, value)
        nn.init.uniform_(self.learnable_tokens_b, -value, value)

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            return self.learnable_tokens_a @ self.learnable_tokens_b
        if layer < 0 or layer >= self.num_layers:
            raise IndexError(
                f"非法 REIN layer={layer}，有效范围为 0～{self.num_layers - 1}。"
            )
        return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]


def count_rein_parameters(module: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": sum(p.numel() for p in module.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in module.parameters() if p.requires_grad
        ),
    }
