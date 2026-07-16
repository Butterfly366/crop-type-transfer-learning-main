"""
将 SSL4EO-S12 MoCo ViT-S/16 + SMP UNet Decoder
接入 crop-type-transfer-learning 的 TorchGeo 训练框架。

设计原则
--------
保留 SemanticSegmentationTask 已实现的：

1. training_step
2. validation_step
3. test_step
4. predict_step
5. 损失函数
6. 评价指标
7. 优化器
8. 学习率调度器

这里只覆盖 configure_models()，替换模型构建逻辑。
"""

from collections.abc import Sequence

from torchgeo.trainers import SemanticSegmentationTask

from models.ssl4eo_moco_vit_unet import SSL4EOMoCoViTUNet


class SSL4EOMoCoViTUNetTask(SemanticSegmentationTask):
    """
    SSL4EO-S12 MoCo ViT-S/16 作物语义分割任务。

    模型结构
    --------
    SSL4EO-S12 MoCo ViT-S/16
        ↓
    抽取 Transformer blocks [1, 4, 7, 9, 11]
        ↓
    多尺度特征适配器
        ↓
    segmentation-models-pytorch 原始 UnetDecoder
        ↓
    逐像元作物类型分类结果

    该类继承 SemanticSegmentationTask，
    因此训练、验证和测试逻辑与原仓库保持一致。
    """

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        selected_blocks: Sequence[int] = (1, 4, 7, 9, 11),
        freeze_vit: bool = True,
        in_channels: int = 13,
        num_classes: int = 6,
        loss: str = "ce",
        class_weights=None,
        ignore_index: int | None = 0,
        lr: float = 1e-3,
        patience: int = 1,
    ) -> None:
        """
        初始化训练任务。

        参数
        ----
        checkpoint_path:
            SSL4EO-S12 MoCo ViT-S/16 官方权重路径。

        image_size:
            输入影像块尺寸。当前模型固定使用 256×256。

        selected_blocks:
            需要抽取的 Transformer block 编号。
            编号从 0 开始。

        freeze_vit:
            True 表示冻结 ViT，只训练：
                - 中间层归一化模块
                - 多尺度适配器
                - UNet decoder
                - segmentation head

        in_channels:
            输入波段数。SSL4EO-S12 L1C 模型固定为 13。

        num_classes:
            输出类别数。原论文配置为 6。

        loss:
            沿用 TorchGeo SemanticSegmentationTask，
            支持 ce、jaccard 和 focal。

        ignore_index:
            标签中需要忽略的类别编号。
            原仓库配置为 0。

        lr:
            初始学习率。

        patience:
            学习率调度器 patience。
        """

        if in_channels != 13:
            raise ValueError(
                "SSL4EO-S12 MoCo ViT-S/16 权重要求 13 波段输入，"
                f"当前设置为 {in_channels}。"
            )

        if image_size != 256:
            raise ValueError(
                "当前特征适配器按照 256×256 输入设计，"
                f"当前设置为 {image_size}。"
            )

        if len(selected_blocks) != 5:
            raise ValueError(
                "需要恰好抽取 5 个 Transformer block，"
                "以生成 UNet 所需的 5 级编码特征。"
            )

        # configure_models() 会在父类初始化期间被调用。
        # 因此必须在 super().__init__() 之前保存这些参数。
        self.ssl4eo_checkpoint_path = checkpoint_path
        self.ssl4eo_image_size = image_size
        self.ssl4eo_selected_blocks = tuple(selected_blocks)
        self.ssl4eo_freeze_vit = freeze_vit

        # 继续使用 SemanticSegmentationTask 的所有训练逻辑。
        #
        # model、backbone 和 weights 在这里仅用于满足父类接口。
        # 真正的模型会在我们覆盖的 configure_models() 中创建。
        super().__init__(
            model="unet",
            backbone="resnet50",
            weights=None,
            in_channels=in_channels,
            num_classes=num_classes,
            num_filters=1,
            loss=loss,
            class_weights=class_weights,
            ignore_index=ignore_index,
            lr=lr,
            patience=patience,
            freeze_backbone=False,
            freeze_decoder=False,
        )

        # 覆盖父类为了兼容接口而保存的错误名称
        self.hparams["model"] = "ssl4eo_moco_vit_unet"
        self.hparams["backbone"] = "ssl4eo_s12_moco_vit_small_patch16"
        self.hparams["weights"] = checkpoint_path
        self.hparams["image_size"] = image_size
        self.hparams["selected_blocks"] = list(selected_blocks)
        self.hparams["freeze_vit"] = freeze_vit
        self.hparams["fine_tuning_method"] = (
            "frozen_vit_train_adapters_decoder"
            if freeze_vit
            else "full_fine_tuning"
        )

        # 比较各种PEFT方法，再记录：
        trainable_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        frozen_parameters = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )

        total_parameters = trainable_parameters + frozen_parameters

        self.hparams["trainable_parameters"] = trainable_parameters
        self.hparams["frozen_parameters"] = frozen_parameters
        self.hparams["total_parameters"] = total_parameters
        self.hparams["trainable_ratio"] = trainable_parameters / total_parameters

        # 将自定义参数写入 Lightning checkpoint 的超参数中，
        # 便于后续使用 ckpt_path 恢复模型。
        self.save_hyperparameters(
            {
                "checkpoint_path": checkpoint_path,
                "image_size": image_size,
                "selected_blocks": list(selected_blocks),
                "freeze_vit": freeze_vit,
            }
        )

    def configure_models(self) -> None:
        """
        创建 SSL4EO-S12 MoCo ViT-S/16 + UNet 模型。

        父类 SemanticSegmentationTask 原本会在这里创建：

            smp.Unet(
                encoder_name="resnet50",
                ...
            )

        当前仅替换这一部分，其余训练流程保持不变。
        """

        self.model = SSL4EOMoCoViTUNet(
            checkpoint_path=self.ssl4eo_checkpoint_path,
            num_classes=int(self.hparams["num_classes"]),
            image_size=self.ssl4eo_image_size,
            freeze_vit=self.ssl4eo_freeze_vit,
        )

    def on_fit_start(self) -> None:
        """在训练开始时记录数据和训练相关配置。"""

        datamodule = self.trainer.datamodule

        batch_size = getattr(datamodule, "batch_size", None)

        experiment_config = {
            "batch_size": batch_size,
        }

        # 写入 Lightning 超参数
        for key, value in experiment_config.items():
            self.hparams[key] = value

        # 写入 WandB / CSVLogger
        if self.logger is not None:
            self.logger.log_hyperparams(experiment_config)