"""
ViT 参数高效微调基础模块。

包含：
1. LoRAQKVLinear：
   包装 timm ViT 中联合的 qkv 线性层，只对 Query 和 Value 添加 LoRA。
2. AdaptFormerMLP：
   包装 timm ViT Block 的原始 MLP，增加并行瓶颈 Adapter 分支。
3. 注入、冻结和参数统计工具。

设计目标：
- 不修改 timm 源码；
- 保留原始 SSL4EO-S12 MoCo ViT 权重；
- 新增分支零初始化，使注入前后初始输出一致；
- 仅让 PEFT 参数参与 ViT 主干微调。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn


class LoRAProjection(nn.Module):
    """
    标准低秩增量分支。

    该模块不包含原始线性层，只计算：

        delta(x) = scaling * B(A(dropout(x)))

    参数
    ----
    in_features:
        输入特征维度。
    out_features:
        输出特征维度。
    rank:
        LoRA 低秩维度 r。
    alpha:
        LoRA 缩放参数 alpha，最终缩放系数为 alpha / rank。
    dropout:
        LoRA 分支输入 dropout。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if rank <= 0:
            raise ValueError(f"LoRA rank 必须大于 0，当前为 {rank}。")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"LoRA dropout 必须位于 [0, 1)，当前为 {dropout}。"
            )

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(
            in_features,
            self.rank,
            bias=False,
        )
        self.lora_B = nn.Linear(
            self.rank,
            out_features,
            bias=False,
        )

        # A 使用 Kaiming 初始化；B 初始化为 0。
        # 因此训练开始时 LoRA 分支输出严格为 0，
        # 模型初始行为与原始预训练 ViT 一致。
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        """计算低秩增量。"""
        return self.lora_B(
            self.lora_A(self.dropout(x))
        ) * self.scaling


class LoRAQKVLinear(nn.Module):
    """
    包装 timm ViT 的联合 qkv 线性层，只适配 Q 和 V。

    timm 的 Attention 通常包含：

        qkv = nn.Linear(embed_dim, 3 * embed_dim)

    输出排列为：

        [Query | Key | Value]

    本模块保留原始 qkv 输出，并分别向 Query 和 Value 段添加
    LoRA 增量，Key 段保持完全不变。

    注意：
    - base_qkv 的参数应被冻结；
    - 本模块保持与 nn.Linear 相同的 forward 输入/输出接口；
    - timm Attention 可以直接继续调用 self.qkv(x)。
    """

    def __init__(
        self,
        base_qkv: nn.Linear,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.0,
        enable_query: bool = True,
        enable_value: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(base_qkv, nn.Linear):
            raise TypeError(
                "LoRAQKVLinear 只能包装 nn.Linear，"
                f"实际类型为 {type(base_qkv).__name__}。"
            )

        if base_qkv.out_features != 3 * base_qkv.in_features:
            raise ValueError(
                "联合 qkv 层应满足 out_features = 3 * in_features，"
                f"当前为 {base_qkv.in_features} -> "
                f"{base_qkv.out_features}。"
            )

        if not enable_query and not enable_value:
            raise ValueError("Query 和 Value 至少需要启用一个 LoRA 分支。")

        self.base_qkv = base_qkv
        self.embed_dim = base_qkv.in_features
        self.enable_query = bool(enable_query)
        self.enable_value = bool(enable_value)

        # 明确冻结原始 qkv。
        for parameter in self.base_qkv.parameters():
            parameter.requires_grad = False

        self.lora_q = (
            LoRAProjection(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            if self.enable_query
            else None
        )

        self.lora_v = (
            LoRAProjection(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            if self.enable_value
            else None
        )

    @property
    def in_features(self) -> int:
        """兼容部分依赖 nn.Linear 属性的外部代码。"""
        return self.base_qkv.in_features

    @property
    def out_features(self) -> int:
        """兼容部分依赖 nn.Linear 属性的外部代码。"""
        return self.base_qkv.out_features

    @property
    def weight(self) -> Tensor:
        """暴露原始 qkv 权重，便于调试和状态检查。"""
        return self.base_qkv.weight

    @property
    def bias(self) -> Tensor | None:
        """暴露原始 qkv 偏置。"""
        return self.base_qkv.bias

    def forward(self, x: Tensor) -> Tensor:
        """
        计算原始 qkv，并向 Q/V 对应区间添加低秩增量。

        输入：
            [..., embed_dim]

        输出：
            [..., 3 * embed_dim]
        """
        qkv = self.base_qkv(x)

        # 不使用原地切片修改，避免潜在的 autograd 视图问题。
        query, key, value = qkv.split(self.embed_dim, dim=-1)

        if self.lora_q is not None:
            query = query + self.lora_q(x)

        if self.lora_v is not None:
            value = value + self.lora_v(x)

        return torch.cat([query, key, value], dim=-1)


class AdaptFormerAdapter(nn.Module):
    """
    AdaptFormer 瓶颈分支。

    结构：

        embed_dim
            -> down_proj
        bottleneck_dim
            -> GELU
            -> dropout
            -> up_proj
        embed_dim

    up_proj 零初始化，使 Adapter 在注入时输出为 0。
    """

    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if bottleneck_dim <= 0:
            raise ValueError(
                "AdaptFormer bottleneck_dim 必须大于 0，"
                f"当前为 {bottleneck_dim}。"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "AdaptFormer dropout 必须位于 [0, 1)，"
                f"当前为 {dropout}。"
            )

        self.down_proj = nn.Linear(
            embed_dim,
            bottleneck_dim,
            bias=True,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(
            bottleneck_dim,
            embed_dim,
            bias=True,
        )

        # down_proj 使用 PyTorch 默认初始化。
        # up_proj 零初始化，保证初始 Adapter 输出为 0。
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """计算 Adapter 分支输出。"""
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return x


class AdaptFormerMLP(nn.Module):
    """
    将 AdaptFormer 作为原始 ViT MLP 的并行分支。

    timm Block 原始逻辑近似为：

        x = x + drop_path(mlp(norm2(x)))

    替换 block.mlp 后变为：

        x = x + drop_path(
            original_mlp(norm2(x))
            + scale * adapter(norm2(x))
        )

    因为包装的是 block.mlp，所以无需复制或修改 timm Block.forward()。
    """

    def __init__(
        self,
        base_mlp: nn.Module,
        embed_dim: int,
        bottleneck_dim: int = 64,
        scale: float = 0.1,
        dropout: float = 0.0,
        learnable_scale: bool = False,
    ) -> None:
        super().__init__()

        self.base_mlp = base_mlp

        # 原始 MLP 属于预训练 ViT，应保持冻结。
        for parameter in self.base_mlp.parameters():
            parameter.requires_grad = False

        self.adapter = AdaptFormerAdapter(
            embed_dim=embed_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )

        if learnable_scale:
            self.scale = nn.Parameter(
                torch.tensor(float(scale), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "scale",
                torch.tensor(float(scale), dtype=torch.float32),
                persistent=True,
            )

    def forward(self, x: Tensor) -> Tensor:
        """返回原始 MLP 与 AdaptFormer 分支之和。"""
        base_output = self.base_mlp(x)
        adapter_output = self.adapter(x)

        # 转换到当前输出 dtype/device，兼容混合精度训练。
        scale = self.scale.to(
            device=adapter_output.device,
            dtype=adapter_output.dtype,
        )

        return base_output + scale * adapter_output


def freeze_module(module: nn.Module) -> None:
    """冻结模块中的全部参数。"""
    for parameter in module.parameters():
        parameter.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """解冻模块中的全部参数。"""
    for parameter in module.parameters():
        parameter.requires_grad = True


def _validate_block_indices(
    num_blocks: int,
    block_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    """检查并标准化需要注入 PEFT 的 block 编号。"""
    if block_indices is None:
        return tuple(range(num_blocks))

    normalized = tuple(int(index) for index in block_indices)

    if not normalized:
        raise ValueError("block_indices 不能为空。")

    if len(set(normalized)) != len(normalized):
        raise ValueError("block_indices 中不能包含重复编号。")

    for index in normalized:
        if index < 0 or index >= num_blocks:
            raise ValueError(
                f"非法 block 编号 {index}，"
                f"有效范围为 0～{num_blocks - 1}。"
            )

    return normalized


def inject_lora_into_vit(
    vit: nn.Module,
    rank: int = 4,
    alpha: float = 4.0,
    dropout: float = 0.0,
    block_indices: Sequence[int] | None = None,
    enable_query: bool = True,
    enable_value: bool = True,
) -> tuple[int, ...]:
    """
    向 timm ViT 的指定 Transformer blocks 注入 Q/V LoRA。

    返回
    ----
    tuple[int, ...]
        实际完成注入的 block 编号。
    """
    if not hasattr(vit, "blocks"):
        raise AttributeError("目标 ViT 不存在 blocks 属性。")

    indices = _validate_block_indices(
        num_blocks=len(vit.blocks),
        block_indices=block_indices,
    )

    for block_index in indices:
        block = vit.blocks[block_index]

        if not hasattr(block, "attn") or not hasattr(block.attn, "qkv"):
            raise AttributeError(
                f"vit.blocks[{block_index}] 不存在 attn.qkv。"
            )

        if isinstance(block.attn.qkv, LoRAQKVLinear):
            raise RuntimeError(
                f"vit.blocks[{block_index}].attn.qkv 已经注入 LoRA。"
            )

        block.attn.qkv = LoRAQKVLinear(
            base_qkv=block.attn.qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            enable_query=enable_query,
            enable_value=enable_value,
        )

    return indices


def inject_adaptformer_into_vit(
    vit: nn.Module,
    bottleneck_dim: int = 64,
    scale: float = 0.1,
    dropout: float = 0.0,
    learnable_scale: bool = False,
    block_indices: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """
    向 timm ViT 的指定 Transformer blocks 注入 AdaptFormer。

    返回
    ----
    tuple[int, ...]
        实际完成注入的 block 编号。
    """
    if not hasattr(vit, "blocks"):
        raise AttributeError("目标 ViT 不存在 blocks 属性。")

    indices = _validate_block_indices(
        num_blocks=len(vit.blocks),
        block_indices=block_indices,
    )

    embed_dim = getattr(vit, "embed_dim", None)

    if embed_dim is None:
        raise AttributeError("目标 ViT 不存在 embed_dim 属性。")

    for block_index in indices:
        block = vit.blocks[block_index]

        if not hasattr(block, "mlp"):
            raise AttributeError(
                f"vit.blocks[{block_index}] 不存在 mlp。"
            )

        if isinstance(block.mlp, AdaptFormerMLP):
            raise RuntimeError(
                f"vit.blocks[{block_index}].mlp 已经注入 AdaptFormer。"
            )

        block.mlp = AdaptFormerMLP(
            base_mlp=block.mlp,
            embed_dim=int(embed_dim),
            bottleneck_dim=bottleneck_dim,
            scale=scale,
            dropout=dropout,
            learnable_scale=learnable_scale,
        )

    return indices


def count_parameters(model: nn.Module) -> dict[str, int | float]:
    """统计总参数、可训练参数和可训练参数比例。"""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen = total - trainable

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_ratio": trainable / total if total > 0 else 0.0,
    }


def iter_trainable_parameter_names(
    model: nn.Module,
) -> Iterable[tuple[str, int]]:
    """逐个返回可训练参数名称及参数量，便于检查冻结是否正确。"""
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            yield name, parameter.numel()
