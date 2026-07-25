"""
批量生成 SSL4EO-S12 MAE ViT-S/16 + GDA-SLR 自监督域适配配置。

生成范围
--------
数据集：
    cdl、eurocrops、nccm、sact、sas

实验规模：
    900ID
    4000OOD-900ID

生成位置：
    experiments/<dataset>/generated_peft/slr_mae/<experiment>.yaml

设计原则
--------
1. 不手工重复每个数据集的数据路径；
2. 优先读取现有 LoRA/AdaptFormer 监督训练 YAML，复用其中的 data 配置；
3. 将监督模型替换为 SSL4EOMAEViTSLRTask；
4. 自监督阶段只使用影像，Task 会忽略标签；
5. 默认使用 GDA 原论文配置：
   - mask_ratio=0.75
   - rank=16
   - patch embedding SLR=True
   - LayerNorm 可训练
   - cls/mask token 可训练
   - loss_on_all_patches=True
   - lr=1e-4
   - batch_size=32
   - warmup_epochs=5
   - max_steps=25000
6. 默认不覆盖已有文件，使用 --overwrite 才会覆盖。

运行示例
--------
python yaml_scripts/generate_ssl4eo_mae_slr_pretrain_yamls.py

仅预览：
python yaml_scripts/generate_ssl4eo_mae_slr_pretrain_yamls.py --dry-run

覆盖已有配置：
python yaml_scripts/generate_ssl4eo_mae_slr_pretrain_yamls.py --overwrite
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml


# 当前脚本位于 yaml_scripts/，项目根目录是其父目录。
REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = (
    "cdl",
    "eurocrops",
    "nccm",
    "sact",
    "sas",
)

EXPERIMENTS = (
    "900ID",
    "4000OOD-900ID",
)

# 优先从已有 PEFT 监督实验中复制 data 配置。
# 按顺序查找，找到第一个存在的文件就使用。
REFERENCE_CONFIG_PATTERNS = (
    "experiments/{dataset}/generated_peft/lora/{experiment}.yaml",
    "experiments/{dataset}/generated_peft/adaptformer/{experiment}.yaml",
    "experiments/{dataset}/PEFT/lora/{experiment}.yaml",
    "experiments/{dataset}/PEFT/adaptformer/{experiment}.yaml",
    "experiments/{dataset}/PEFT/{experiment}.yaml",
)

MAE_CHECKPOINT = (
    "/home/hzm/.cache/torch/hub/checkpoints/ssl4eo/"
    "B13_vits16_mae_0099_ckpt.pth"
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "批量生成 SSL4EO MAE ViT-S/16 + GDA-SLR "
            "自监督训练 YAML。"
        )
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="需要生成配置的数据集。",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=list(EXPERIMENTS),
        help="需要生成的实验规模。",
    )
    parser.add_argument(
        "--checkpoint",
        default=MAE_CHECKPOINT,
        help="SSL4EO-S12 MAE ViT-S/16 完整 checkpoint 路径。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的输出 YAML。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印生成计划，不写文件。",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML，并确保根节点是字典。"""

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        raise ValueError(f"YAML 根节点不是映射：{path}")

    return value


def find_reference_config(
    dataset: str,
    experiment: str,
) -> Path:
    """
    查找已有监督训练 YAML。

    该 YAML 只用于继承：
    - data.class_path
    - data.init_args
    - 部分 trainer/logger 风格

    模型配置会被完全替换。
    """

    checked: list[Path] = []

    for pattern in REFERENCE_CONFIG_PATTERNS:
        candidate = REPO_ROOT / pattern.format(
            dataset=dataset,
            experiment=experiment,
        )
        checked.append(candidate)

        if candidate.is_file():
            return candidate

    checked_text = "\n".join(f"  - {path}" for path in checked)

    raise FileNotFoundError(
        f"找不到 {dataset}/{experiment} 的参考监督 YAML。\n"
        f"已检查：\n{checked_text}"
    )


def update_batch_size(
    data_config: dict[str, Any],
    batch_size: int,
) -> None:
    """
    将 DataModule 的 batch_size 改为 GDA 默认值。

    当前仓库的 DataModule 通常使用：
        data.init_args.batch_size
    """

    init_args = data_config.setdefault("init_args", {})

    if not isinstance(init_args, dict):
        raise ValueError("data.init_args 必须是字典。")

    init_args["batch_size"] = int(batch_size)


def build_logger_config(
    dataset: str,
    experiment: str,
    reference: dict[str, Any],
) -> dict[str, Any]:
    """
    继承现有 logger 类，但重写 project/name/save_dir。

    这样可以继续使用仓库中的 WandbLoggerExplicit，
    同时避免自监督实验与监督实验混在一起。
    """

    reference_logger = (
        reference.get("trainer", {})
        .get("logger", {})
    )

    if not isinstance(reference_logger, dict):
        reference_logger = {}

    logger = copy.deepcopy(reference_logger)

    # 如果参考 YAML 没有 logger，则使用仓库常用 logger。
    if "class_path" not in logger:
        logger["class_path"] = (
            "custom_loggers.wandb_explicit."
            "WandbLoggerExplicit"
        )

    init_args = logger.setdefault("init_args", {})

    if not isinstance(init_args, dict):
        init_args = {}
        logger["init_args"] = init_args

    project_prefix = (
        dataset.upper()
        if dataset != "eurocrops"
        else "EuroCrops"
    )

    init_args["project"] = (
        f"{project_prefix}-{experiment}-SLR-MAE"
    )
    init_args["name"] = (
        f"{project_prefix}-{experiment}-"
        "SSL4EO-MAE-ViTS16-SLR-r16-Pretrain"
    )
    init_args["save_dir"] = (
        f"./outputs/{dataset}/PEFT/slr_mae"
    )

    # 保留参考配置中的 online/offline 设置；
    # 若没有，则使用在线记录。
    init_args.setdefault("offline", False)
    init_args.setdefault("log_model", False)

    return logger


def build_callbacks() -> list[dict[str, Any]]:
    """建立以 val_loss 为监控指标的 checkpoint 配置。"""

    return [
        {
            "class_path": (
                "lightning.pytorch.callbacks.ModelCheckpoint"
            ),
            "init_args": {
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 1,
                "save_last": True,
                "filename": (
                    "epoch{epoch:02d}-"
                    "step{step}-"
                    "val_loss{val_loss:.6f}"
                ),
            },
        }
    ]


def build_config(
    dataset: str,
    experiment: str,
    reference_path: Path,
    checkpoint_path: str,
) -> dict[str, Any]:
    """根据参考监督 YAML 构建自监督 SLR-MAE YAML。"""

    reference = load_yaml(reference_path)

    data_config = copy.deepcopy(reference.get("data"))

    if not isinstance(data_config, dict):
        raise ValueError(
            f"参考 YAML 缺少有效 data 配置：{reference_path}"
        )

    update_batch_size(
        data_config=data_config,
        batch_size=32,
    )

    output_root = (
        f"./outputs/{dataset}/PEFT/slr_mae/"
        f"{experiment}/ssl_pretrain"
    )

    config: dict[str, Any] = {
        "seed_everything": 0,
        "trainer": {
            # GDA 原实现以 step 数控制训练长度。
            "max_steps": 25000,
            # 使用 max_steps 时，将 max_epochs 设为 -1，
            # 避免 epoch 上限提前终止。
            "max_epochs": -1,
            "accelerator": "gpu",
            "devices": 1,
            "logger": build_logger_config(
                dataset=dataset,
                experiment=experiment,
                reference=reference,
            ),
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 1,
            "enable_checkpointing": True,
            "default_root_dir": output_root,
            "callbacks": build_callbacks(),
        },
        "model": {
            "class_path": (
                "trainers.ssl4eo_mae_vit_slr_task."
                "SSL4EOMAEViTSLRTask"
            ),
            "init_args": {
                "checkpoint_path": checkpoint_path,
                "image_size": 256,
                "in_channels": 13,
                "patch_size": 16,
                "mask_ratio": 0.75,
                "slr_rank": 16,
                "slr_blocks": None,
                "patch_embed_adapter": True,
                "norm_trainable": True,
                "train_cls_mask_tokens": True,
                "loss_on_all_patches": True,
                "norm_pix_loss": False,
                "decoder_embed_dim": 512,
                "decoder_depth": 8,
                "decoder_num_heads": 16,
                "lr": 0.0001,
                "weight_decay": 0.05,
                "min_lr": 0.0,
                "warmup_epochs": 5,
            },
        },
        "data": data_config,
    }

    return config


def output_path(
    dataset: str,
    experiment: str,
) -> Path:
    """返回目标 YAML 路径。"""

    return (
        REPO_ROOT
        / "experiments"
        / dataset
        / "generated_peft"
        / "slr_mae"
        / f"{experiment}.yaml"
    )


def write_yaml(
    path: Path,
    config: dict[str, Any],
    overwrite: bool,
    dry_run: bool,
) -> str:
    """
    写出 YAML。

    返回状态：
    - CREATED
    - OVERWRITTEN
    - SKIPPED
    - DRY-RUN
    """

    existed = path.exists()

    if existed and not overwrite:
        return "SKIPPED"

    if dry_run:
        return "DRY-RUN"

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=100,
        )

    return "OVERWRITTEN" if existed else "CREATED"


def validate_generated_config(
    config: dict[str, Any],
    dataset: str,
    experiment: str,
) -> None:
    """在写出前做关键字段检查。"""

    model = config.get("model", {})
    init_args = model.get("init_args", {})
    trainer = config.get("trainer", {})
    data = config.get("data", {})
    data_args = data.get("init_args", {})

    expected_class = (
        "trainers.ssl4eo_mae_vit_slr_task."
        "SSL4EOMAEViTSLRTask"
    )

    if model.get("class_path") != expected_class:
        raise ValueError(
            f"{dataset}/{experiment}: model.class_path 错误。"
        )

    required_model_values = {
        "image_size": 256,
        "in_channels": 13,
        "patch_size": 16,
        "mask_ratio": 0.75,
        "slr_rank": 16,
        "patch_embed_adapter": True,
        "norm_trainable": True,
        "train_cls_mask_tokens": True,
        "loss_on_all_patches": True,
        "lr": 0.0001,
        "warmup_epochs": 5,
    }

    for key, expected in required_model_values.items():
        actual = init_args.get(key)

        if actual != expected:
            raise ValueError(
                f"{dataset}/{experiment}: "
                f"{key}={actual!r}，预期 {expected!r}。"
            )

    if trainer.get("max_steps") != 25000:
        raise ValueError(
            f"{dataset}/{experiment}: max_steps 不是 25000。"
        )

    if data_args.get("batch_size") != 32:
        raise ValueError(
            f"{dataset}/{experiment}: batch_size 不是 32。"
        )


def main() -> int:
    """生成全部自监督训练配置。"""

    args = parse_args()

    created = 0
    overwritten = 0
    skipped = 0
    failed = 0

    print("=" * 100)
    print("SSL4EO MAE ViT-S/16 + GDA-SLR 自监督 YAML 生成")
    print("项目根目录：", REPO_ROOT)
    print("Checkpoint：", args.checkpoint)
    print("数据集：", ", ".join(args.datasets))
    print("实验：", ", ".join(args.experiments))
    print("dry-run：", args.dry_run)
    print("overwrite：", args.overwrite)
    print("=" * 100)

    for dataset in args.datasets:
        for experiment in args.experiments:
            target = output_path(
                dataset=dataset,
                experiment=experiment,
            )

            try:
                reference = find_reference_config(
                    dataset=dataset,
                    experiment=experiment,
                )

                config = build_config(
                    dataset=dataset,
                    experiment=experiment,
                    reference_path=reference,
                    checkpoint_path=args.checkpoint,
                )

                validate_generated_config(
                    config=config,
                    dataset=dataset,
                    experiment=experiment,
                )

                status = write_yaml(
                    path=target,
                    config=config,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )

                print(
                    f"[{status:11s}] "
                    f"{dataset:10s} "
                    f"{experiment:15s} "
                    f"reference={reference.relative_to(REPO_ROOT)} "
                    f"-> {target.relative_to(REPO_ROOT)}"
                )

                if status == "CREATED":
                    created += 1
                elif status == "OVERWRITTEN":
                    overwritten += 1
                elif status == "SKIPPED":
                    skipped += 1

            except Exception as exc:  # noqa: BLE001
                failed += 1

                print(
                    f"[FAILED     ] "
                    f"{dataset:10s} "
                    f"{experiment:15s} "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    print("=" * 100)
    print(
        f"完成：created={created}, "
        f"overwritten={overwritten}, "
        f"skipped={skipped}, "
        f"failed={failed}"
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())