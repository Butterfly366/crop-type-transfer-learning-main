"""
SSL4EO-S12 MoCo ViT-S/16 + 原始 SMP UNet Decoder。

整体结构
--------
输入：
    [B, 13, 256, 256]

ViT 中间层：
    blocks [1, 4, 7, 9, 11]
    注意：这里采用 Python 的从 0 开始编号。

每个 Transformer block 输出：
    [B, 257, 384]

去除 CLS token 后：
    [B, 256, 384]

恢复为空间特征：
    [B, 384, 16, 16]

经过五个独立尺度适配器后：
    feature[0] = [B,   13, 256, 256]  原始输入
    feature[1] = [B,   64, 128, 128]
    feature[2] = [B,  256,  64,  64]
    feature[3] = [B,  512,  32,  32]
    feature[4] = [B, 1024,  16,  16]
    feature[5] = [B, 2048,   8,   8]

这与原始 ResNet50 encoder 给 SMP UnetDecoder 的接口一致。
"""

from pathlib import Path
from typing import Sequence

import segmentation_models_pytorch as smp
import timm
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ScaleAdapter(nn.Module):
    """
    将 ViT 的单尺度特征转换为指定通道数和空间尺寸。

    输入：
        [B, 384, 16, 16]

    输出示例：
        [B, 64, 128, 128]
        [B, 256, 64, 64]
        ...
        [B, 2048, 8, 8]

    处理顺序：
        1. 先在 16×16 上使用 1×1 卷积调整通道；
        2. 再插值到目标空间尺寸；
        3. 使用逐通道 3×3 卷积进行局部空间细化。

    先降维再上采样可以明显降低高分辨率特征的显存占用。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        target_size: tuple[int, int],
    ) -> None:
        super().__init__()

        self.target_size = target_size

        # 1×1 卷积负责通道映射。
        self.channel_projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            # GroupNorm 不依赖 batch size，适合少样本和小 batch 训练。
            nn.GroupNorm(
                num_groups=self._choose_num_groups(out_channels),
                num_channels=out_channels,
            ),
            nn.GELU(),
        )

        # 逐通道卷积只处理空间邻域，不产生巨大的参数量。
        self.spatial_refinement = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=self._choose_num_groups(out_channels),
                num_channels=out_channels,
            ),
            nn.GELU(),
        )

    @staticmethod
    def _choose_num_groups(channels: int) -> int:
        """
        为 GroupNorm 选择可以整除通道数的组数。

        当前目标通道数均可以被 32 整除，但保留通用处理。
        """

        for groups in (32, 16, 8, 4, 2, 1):
            if channels % groups == 0:
                return groups

        return 1

    def forward(self, x: Tensor) -> Tensor:
        """执行通道映射、尺度调整和空间细化。"""

        x = self.channel_projection(x)

        if x.shape[-2:] != self.target_size:
            x = F.interpolate(
                x,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )

        x = self.spatial_refinement(x)

        return x


class SSL4EOMoCoViTS16Encoder(nn.Module):
    """
    兼容 SMP UNet encoder 接口的 SSL4EO-S12 MoCo ViT-S/16。

    该编码器：
        1. 创建 13 波段 ViT-S/16；
        2. 加载 SSL4EO-S12 MoCo 权重；
        3. 将位置编码从 224 输入对应的 14×14，
           插值到 256 输入对应的 16×16；
        4. 抽取 blocks [1, 4, 7, 9, 11]；
        5. 输出与原 ResNet50 encoder 相同的六级特征列表。
    """

    # SMP decoder 根据这个属性建立各级跳跃连接。
    out_channels = [13, 64, 256, 512, 1024, 2048]

    # 最深特征相对输入缩小 32 倍：256 → 8。
    output_stride = 32

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        selected_blocks: Sequence[int] = (1, 4, 7, 9, 11),
        freeze_vit: bool = True,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = 16
        self.embed_dim = 384
        self._in_channels = 13
        self._depth = 5

        self.selected_blocks = tuple(selected_blocks)

        self._validate_selected_blocks()

        # 创建与官方权重完全匹配的 ViT-S/16。
        #
        # img_size=256 会使模型自身建立：
        #     pos_embed = [1, 257, 384]
        #
        # 后面将 checkpoint 中的 [1, 197, 384]
        # 插值为 [1, 257, 384] 后再加载。
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            in_chans=13,
            num_classes=0,
            img_size=image_size,
        )

        self._load_ssl4eo_moco_checkpoint(
            checkpoint_path=checkpoint_path
        )

        # 对每个抽取层分别进行 LayerNorm。
        #
        # ViT 的中间 block 输出没有全部经过最终 self.vit.norm，
        # 因而给每一级设置独立归一化，使不同深度特征的尺度更稳定。
        self.feature_norms = nn.ModuleList(
            [
                nn.LayerNorm(self.embed_dim)
                for _ in self.selected_blocks
            ]
        )

        # 五个独立尺度适配模块。
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

        if freeze_vit:
            self.freeze_vit_backbone()

    def _validate_selected_blocks(self) -> None:
        """检查抽取层编号是否合法。"""

        if len(self.selected_blocks) != 5:
            raise ValueError(
                "UNet 需要 5 级编码特征，"
                "selected_blocks 必须包含 5 个 block 编号。"
            )

        if tuple(sorted(self.selected_blocks)) != self.selected_blocks:
            raise ValueError(
                "selected_blocks 必须按照从浅到深的顺序排列。"
            )

        if len(set(self.selected_blocks)) != len(self.selected_blocks):
            raise ValueError(
                "selected_blocks 中不能存在重复编号。"
            )

        for block_index in self.selected_blocks:
            if block_index < 0 or block_index >= 12:
                raise ValueError(
                    f"非法 block 编号：{block_index}。"
                    "ViT-S/16 的有效编号为 0～11。"
                )

    def freeze_vit_backbone(self) -> None:
        """
        冻结 SSL4EO-S12 ViT 主干。

        特征归一化层、尺度适配器和 UNet decoder 仍保持可训练。
        """

        for parameter in self.vit.parameters():
            parameter.requires_grad = False

    def unfreeze_vit_backbone(self) -> None:
        """解冻 ViT 主干，用于全量微调。"""

        for parameter in self.vit.parameters():
            parameter.requires_grad = True

    @staticmethod
    def _extract_backbone_state_dict(
        checkpoint_path: Path,
    ) -> dict[str, Tensor]:
        """
        从官方 MoCo checkpoint 中提取纯 ViT backbone 参数。

        原始参数：
            module.base_encoder.cls_token
            module.base_encoder.blocks.0.attn.qkv.weight
            module.base_encoder.head.0.weight

        转换后：
            cls_token
            blocks.0.attn.qkv.weight

        module.base_encoder.head.* 是 MoCo projection head，
        不属于下游 ViT backbone，因此删除。
        """

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "SSL4EO checkpoint 顶层对象必须是字典。"
            )

        if "state_dict" not in checkpoint:
            raise KeyError(
                "SSL4EO checkpoint 中不存在 state_dict。"
            )

        source_state_dict = checkpoint["state_dict"]
        prefix = "module.base_encoder."

        backbone_state_dict: dict[str, Tensor] = {}

        for key, value in source_state_dict.items():
            if not key.startswith(prefix):
                continue

            new_key = key[len(prefix):]

            # 删除 MoCo 自监督投影头。
            if new_key.startswith("head."):
                continue

            backbone_state_dict[new_key] = value

        if not backbone_state_dict:
            raise RuntimeError(
                "没有从 checkpoint 中提取到 ViT backbone 参数。"
            )

        return backbone_state_dict

    @staticmethod
    def _interpolate_position_embedding(
        state_dict: dict[str, Tensor],
        model: nn.Module,
    ) -> dict[str, Tensor]:
        """
        将 checkpoint 位置编码调整为当前输入尺寸。

        原始：
            [1, 197, 384]
            = 1 个 CLS + 14×14 patch

        目标：
            [1, 257, 384]
            = 1 个 CLS + 16×16 patch
        """

        source_pos_embed = state_dict["pos_embed"]
        target_pos_embed = model.pos_embed

        if source_pos_embed.shape == target_pos_embed.shape:
            return state_dict

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

        state_dict["pos_embed"] = torch.cat(
            [cls_position, patch_positions],
            dim=1,
        )

        return state_dict

    def _load_ssl4eo_moco_checkpoint(
        self,
        checkpoint_path: str,
    ) -> None:
        """读取、转换并严格加载 SSL4EO-S12 MoCo 权重。"""

        path = Path(checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(
                f"找不到 SSL4EO 权重：{path}"
            )

        state_dict = self._extract_backbone_state_dict(path)

        state_dict = self._interpolate_position_embedding(
            state_dict=state_dict,
            model=self.vit,
        )

        load_result = self.vit.load_state_dict(
            state_dict,
            strict=False,
        )

        if load_result.missing_keys:
            raise RuntimeError(
                "ViT 权重存在缺失参数：\n"
                + "\n".join(load_result.missing_keys)
            )

        if load_result.unexpected_keys:
            raise RuntimeError(
                "ViT 权重存在多余参数：\n"
                + "\n".join(load_result.unexpected_keys)
            )

    def _prepare_tokens(self, x: Tensor) -> Tensor:
        """
        执行 ViT block 之前的 patch embedding 和位置编码处理。

        这部分对应 timm VisionTransformer.forward_features()
        在进入 self.blocks 前的主要步骤。
        """

        # [B, 13, 256, 256]
        # → [B, 256, 384]
        x = self.vit.patch_embed(x)

        # 加入 CLS token 和位置编码：
        # [B, 256, 384] → [B, 257, 384]
        x = self.vit._pos_embed(x)

        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        return x

    def _tokens_to_feature_map(
        self,
        tokens: Tensor,
    ) -> Tensor:
        """
        删除 CLS token，并将 patch token 恢复为空间特征。

        [B, 257, 384]
            ↓ 删除第一个 CLS token
        [B, 256, 384]
            ↓ reshape
        [B, 16, 16, 384]
            ↓ permute
        [B, 384, 16, 16]
        """

        if tokens.ndim != 3:
            raise ValueError(
                "Transformer token 应为三维张量 [B, N, C]，"
                f"实际为 {tuple(tokens.shape)}。"
            )

        patch_tokens = tokens[:, 1:, :]

        grid_height = self.image_size // self.patch_size
        grid_width = self.image_size // self.patch_size
        expected_patches = grid_height * grid_width

        if patch_tokens.shape[1] != expected_patches:
            raise RuntimeError(
                "Patch token 数量错误："
                f"期望 {expected_patches}，"
                f"实际 {patch_tokens.shape[1]}。"
            )

        feature_map = patch_tokens.reshape(
            patch_tokens.shape[0],
            grid_height,
            grid_width,
            self.embed_dim,
        )

        feature_map = feature_map.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        return feature_map

    def forward(self, x: Tensor) -> list[Tensor]:
        """
        返回与 SMP ResNetEncoder 一致形式的多尺度特征列表。
        """

        if x.ndim != 4:
            raise ValueError(
                f"输入应为 [B,C,H,W]，实际为 {tuple(x.shape)}。"
            )

        if x.shape[1] != 13:
            raise ValueError(
                f"输入应有 13 个波段，实际为 {x.shape[1]}。"
            )

        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                "当前编码器固定处理 "
                f"{self.image_size}×{self.image_size} 输入，"
                f"实际输入为 {tuple(x.shape[-2:])}。"
            )

        # feature[0] 直接保留原始输入，
        # 与 SMP ResNetEncoder 的第一个返回值一致。
        output_features: list[Tensor] = [x]

        tokens = self._prepare_tokens(x)

        selected_tokens: list[Tensor] = []
        selected_set = set(self.selected_blocks)

        # 逐层执行 Transformer block，
        # 在指定层结束后保存输出。
        for block_index, block in enumerate(self.vit.blocks):
            tokens = block(tokens)

            if block_index in selected_set:
                selected_tokens.append(tokens)

        if len(selected_tokens) != 5:
            raise RuntimeError(
                "中间特征抽取数量不正确："
                f"期望 5，实际 {len(selected_tokens)}。"
            )

        # 对五个不同深度特征分别归一化、恢复空间网格并适配尺度。
        for tokens_i, norm_i, adapter_i in zip(
            selected_tokens,
            self.feature_norms,
            self.adapters,
        ):
            tokens_i = norm_i(tokens_i)
            feature_map = self._tokens_to_feature_map(tokens_i)
            adapted_feature = adapter_i(feature_map)
            output_features.append(adapted_feature)

        return output_features


class SSL4EOMoCoViTUNet(nn.Module):
    """
    完整语义分割模型：

        SSL4EO-S12 MoCo ViT-S/16 encoder
        + 原始 segmentation-models-pytorch UnetDecoder
        + 原始 SMP SegmentationHead
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 6,
        image_size: int = 256,
        freeze_vit: bool = True,
    ) -> None:
        super().__init__()

        # 先创建与原仓库完全相同 decoder 配置的 ResNet50 UNet。
        #
        # encoder_weights=None：
        #     不加载 ImageNet 权重。
        #
        # encoder_depth=5：
        #     生成 5 个 decoder block。
        base_unet = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=13,
            classes=num_classes,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
            activation=None,
        )

        # 替换原 ResNet50 encoder。
        self.encoder = SSL4EOMoCoViTS16Encoder(
            checkpoint_path=checkpoint_path,
            image_size=image_size,
            selected_blocks=(1, 4, 7, 9, 11),
            freeze_vit=freeze_vit,
        )

        # 直接复用 SMP 创建出的原始 UNet decoder 和 segmentation head。
        self.decoder = base_unet.decoder
        self.segmentation_head = base_unet.segmentation_head

    def forward(self, x: Tensor) -> Tensor:
        """完成编码、UNet 解码和分割预测。"""

        features = self.encoder(x)

        # 当前安装的 SMP 版本中，UnetDecoder 接收解包后的多尺度特征。
        decoder_output = self.decoder(features)

        masks = self.segmentation_head(decoder_output)

        return masks
