from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn


class ScaledLowRankLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 16) -> None:
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError(
                "ScaledLowRankLinear 只能包装 nn.Linear，"
                f"实际类型为 {type(base_linear).__name__}。"
            )
        if rank <= 0:
            raise ValueError(f"SLR rank 必须大于 0，当前为 {rank}。")

        self.rank = int(rank)
        self.base_linear = base_linear
        self.in_features = int(base_linear.in_features)
        self.out_features = int(base_linear.out_features)

        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False

        self.in_scaler = nn.Parameter(torch.ones(self.in_features))
        self.out_scaler = nn.Parameter(torch.ones(self.out_features))

        self.down = nn.Linear(self.in_features, self.rank, bias=True)
        self.up = nn.Linear(self.rank, self.out_features, bias=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.in_scaler)
        nn.init.ones_(self.out_scaler)
        nn.init.normal_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    @property
    def weight(self) -> Tensor:
        return self.base_linear.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base_linear.bias

    def forward(self, x: Tensor) -> Tensor:
        x_scaled = x * self.in_scaler
        low_rank_output = self.up(self.down(x_scaled))
        base_output = self.base_linear(x_scaled)
        return (base_output + low_rank_output) * self.out_scaler


def _validate_block_indices(
    num_blocks: int,
    block_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if block_indices is None:
        return tuple(range(num_blocks))

    normalized = tuple(int(index) for index in block_indices)

    if not normalized:
        raise ValueError("slr_blocks 不能为空。")
    if len(set(normalized)) != len(normalized):
        raise ValueError("slr_blocks 中不能包含重复编号。")

    for index in normalized:
        if index < 0 or index >= num_blocks:
            raise ValueError(
                f"非法 block 编号 {index}，"
                f"有效范围为 0～{num_blocks - 1}。"
            )

    return normalized


def inject_slr_into_vit(
    vit: nn.Module,
    rank: int = 16,
    block_indices: Sequence[int] | None = None,
    adapt_qkv: bool = True,
    adapt_attn_proj: bool = True,
    adapt_mlp_fc1: bool = True,
    adapt_mlp_fc2: bool = True,
) -> tuple[int, ...]:
    if not hasattr(vit, "blocks"):
        raise AttributeError("vit 不包含 blocks，无法注入 SLR。")

    selected = _validate_block_indices(
        len(vit.blocks),
        block_indices,
    )

    if not any(
        (adapt_qkv, adapt_attn_proj, adapt_mlp_fc1, adapt_mlp_fc2)
    ):
        raise ValueError("至少需要启用一个 SLR 注入位置。")

    for block_index in selected:
        block = vit.blocks[block_index]

        if adapt_qkv:
            if isinstance(block.attn.qkv, ScaledLowRankLinear):
                raise RuntimeError(
                    f"block {block_index}.attn.qkv 已经注入 SLR。"
                )
            block.attn.qkv = ScaledLowRankLinear(
                block.attn.qkv,
                rank=rank,
            )

        if adapt_attn_proj:
            if isinstance(block.attn.proj, ScaledLowRankLinear):
                raise RuntimeError(
                    f"block {block_index}.attn.proj 已经注入 SLR。"
                )
            block.attn.proj = ScaledLowRankLinear(
                block.attn.proj,
                rank=rank,
            )

        if adapt_mlp_fc1:
            if isinstance(block.mlp.fc1, ScaledLowRankLinear):
                raise RuntimeError(
                    f"block {block_index}.mlp.fc1 已经注入 SLR。"
                )
            block.mlp.fc1 = ScaledLowRankLinear(
                block.mlp.fc1,
                rank=rank,
            )

        if adapt_mlp_fc2:
            if isinstance(block.mlp.fc2, ScaledLowRankLinear):
                raise RuntimeError(
                    f"block {block_index}.mlp.fc2 已经注入 SLR。"
                )
            block.mlp.fc2 = ScaledLowRankLinear(
                block.mlp.fc2,
                rank=rank,
            )

    return selected


def iter_slr_modules(
    module: nn.Module,
) -> Iterable[tuple[str, ScaledLowRankLinear]]:
    for name, child in module.named_modules():
        if isinstance(child, ScaledLowRankLinear):
            yield name, child


def extract_slr_state_dict(module: nn.Module) -> dict[str, Tensor]:
    full_state = module.state_dict()
    prefixes = tuple(
        f"{name}."
        for name, _ in iter_slr_modules(module)
    )

    result: dict[str, Tensor] = {}
    for key, value in full_state.items():
        if not key.startswith(prefixes):
            continue
        if ".base_linear." in key:
            continue
        result[key] = value.detach().cpu().clone()

    if not result:
        raise RuntimeError("模型中没有可导出的 SLR 参数。")

    return result


def load_slr_state_dict(
    module: nn.Module,
    state_dict: dict[str, Tensor],
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    expected = set(extract_slr_state_dict(module).keys())
    provided = set(state_dict.keys())

    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)

    if strict and (missing or unexpected):
        parts = []
        if missing:
            parts.append("缺失 SLR 参数：\n" + "\n".join(missing))
        if unexpected:
            parts.append("多余 SLR 参数：\n" + "\n".join(unexpected))
        raise RuntimeError("\n\n".join(parts))

    current = module.state_dict()
    for key in expected & provided:
        current[key] = state_dict[key]

    module.load_state_dict(current, strict=True)
    return missing, unexpected


def count_slr_parameters(module: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )
    slr = 0

    for _, slr_module in iter_slr_modules(module):
        slr += sum(
            p.numel()
            for name, p in slr_module.named_parameters()
            if p.requires_grad
            and not name.startswith("base_linear.")
        )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "slr_parameters": slr,
    }
