"""
批量生成 SSL4EO-S12 MAE ViT-S/16 + GDA-SLR + UNet 监督训练 YAML。

输出：
    experiments/<dataset>/generated_peft/slr_supervised/<experiment>.yaml

默认范围：
    数据集：cdl、eurocrops、nccm、sact、sas
    实验：900ID、4000OOD-900ID

规则：
1. 从现有 LoRA/AdaptFormer 监督 YAML 复制 data 配置；
2. 自动搜索对应第一阶段 SLR-MAE checkpoint；
3. 优先选择 val_loss 最小的非 last checkpoint；
4. 尚未完成预训练的实验跳过；
5. 已有 YAML 默认不覆盖。
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = ("cdl", "eurocrops", "nccm", "sact", "sas")
EXPERIMENTS = ("900ID", "4000OOD-900ID")

REFERENCE_CONFIG_PATTERNS = (
    "experiments/{dataset}/generated_peft/lora/{experiment}.yaml",
    "experiments/{dataset}/generated_peft/adaptformer/{experiment}.yaml",
    "experiments/{dataset}/PEFT/lora/{experiment}.yaml",
    "experiments/{dataset}/PEFT/adaptformer/{experiment}.yaml",
)

DEFAULT_MAE_CHECKPOINT = (
    "/home/hzm/.cache/torch/hub/checkpoints/ssl4eo/"
    "B13_vits16_mae_0099_ckpt.pth"
)

DISPLAY_NAMES = {
    "cdl": "CDL",
    "eurocrops": "EuroCrops",
    "nccm": "NCCM",
    "sact": "SACT",
    "sas": "SAS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=list(EXPERIMENTS),
    )
    parser.add_argument(
        "--mae-checkpoint",
        default=DEFAULT_MAE_CHECKPOINT,
    )
    parser.add_argument(
        "--checkpoint-map",
        action="append",
        default=[],
        metavar="DATASET:EXPERIMENT:PATH",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="覆盖参考 YAML 的 batch_size；默认保持不变。",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=5.0e-4,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML 根节点不是映射：{path}")
    return value


def find_reference_config(dataset: str, experiment: str) -> Path:
    checked: list[Path] = []
    for pattern in REFERENCE_CONFIG_PATTERNS:
        path = REPO_ROOT / pattern.format(
            dataset=dataset,
            experiment=experiment,
        )
        checked.append(path)
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"找不到 {dataset}/{experiment} 的参考 YAML：\n"
        + "\n".join(f"  - {path}" for path in checked)
    )


def parse_checkpoint_map(
    values: list[str],
) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for raw in values:
        parts = raw.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "--checkpoint-map 必须是 DATASET:EXPERIMENT:PATH"
            )
        dataset, experiment, raw_path = parts
        if dataset not in DATASETS:
            raise ValueError(f"未知数据集：{dataset}")
        if experiment not in EXPERIMENTS:
            raise ValueError(f"未知实验：{experiment}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint 不存在：{path}")
        result[(dataset, experiment)] = path
    return result


def extract_val_loss(path: Path) -> float | None:
    patterns = (
        r"val_loss(?:val_loss)?=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        r"val_loss([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, path.name, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def checkpoint_matches(
    path: Path,
    dataset: str,
    experiment: str,
) -> bool:
    text = str(path).lower()
    return (
        experiment.lower() in text
        and (
            dataset.lower() in text
            or DISPLAY_NAMES[dataset].lower() in text
        )
    )


def find_slr_checkpoint(
    dataset: str,
    experiment: str,
) -> Path | None:
    root = (
        REPO_ROOT
        / "outputs"
        / dataset
        / "PEFT"
        / "slr_mae"
    )
    if not root.is_dir():
        return None

    candidates = [
        path
        for path in root.rglob("*.ckpt")
        if path.is_file()
        and checkpoint_matches(path, dataset, experiment)
    ]
    if not candidates:
        return None

    non_last = [p for p in candidates if p.name != "last.ckpt"]
    with_loss = [
        (loss, path)
        for path in non_last
        if (loss := extract_val_loss(path)) is not None
    ]
    if with_loss:
        with_loss.sort(
            key=lambda item: (
                item[0],
                -item[1].stat().st_mtime,
            )
        )
        return with_loss[0][1].resolve()

    if non_last:
        return max(
            non_last,
            key=lambda path: path.stat().st_mtime,
        ).resolve()

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    ).resolve()


def build_logger(
    dataset: str,
    experiment: str,
    reference: dict[str, Any],
) -> dict[str, Any]:
    logger = copy.deepcopy(
        reference.get("trainer", {}).get("logger", {})
    )
    if not isinstance(logger, dict):
        logger = {}

    logger.setdefault(
        "class_path",
        "custom_loggers.wandb_explicit.WandbLoggerExplicit",
    )
    init_args = logger.setdefault("init_args", {})
    if not isinstance(init_args, dict):
        init_args = {}
        logger["init_args"] = init_args

    display = DISPLAY_NAMES[dataset]
    init_args["project"] = (
        f"{display}-{experiment}-SLR-Supervised"
    )
    init_args["name"] = (
        f"{display}-{experiment}-"
        "SSL4EO-MAE-ViTS16-SLR-r16-UNet-Train"
    )
    init_args["save_dir"] = (
        f"./outputs/{dataset}/PEFT/slr_supervised"
    )
    init_args.setdefault("offline", False)
    init_args.setdefault("log_model", False)
    return logger


def build_config(
    dataset: str,
    experiment: str,
    reference_path: Path,
    mae_checkpoint: Path,
    slr_checkpoint: Path,
    batch_size: int | None,
    max_epochs: int,
    lr: float,
    head_lr: float,
) -> dict[str, Any]:
    reference = load_yaml(reference_path)
    data = copy.deepcopy(reference.get("data"))
    if not isinstance(data, dict):
        raise ValueError(f"参考 YAML 缺少 data：{reference_path}")

    data_init = data.setdefault("init_args", {})
    if not isinstance(data_init, dict):
        raise ValueError("data.init_args 必须是字典。")
    if batch_size is not None:
        data_init["batch_size"] = batch_size

    return {
        "seed_everything": 0,
        "trainer": {
            "max_epochs": max_epochs,
            "accelerator": "gpu",
            "devices": 1,
            "logger": build_logger(
                dataset=dataset,
                experiment=experiment,
                reference=reference,
            ),
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 1,
            "enable_checkpointing": True,
            "default_root_dir": (
                f"./outputs/{dataset}/PEFT/"
                f"slr_supervised/{experiment}"
            ),
            "callbacks": [
                {
                    "class_path": (
                        "lightning.pytorch.callbacks."
                        "ModelCheckpoint"
                    ),
                    "init_args": {
                        "monitor": "val_average_F1-score",
                        "mode": "max",
                        "save_top_k": 1,
                        "save_last": True,
                        "filename": (
                            "epoch{epoch:02d}-"
                            "val_f1{val_average_F1-score:.4f}"
                        ),
                        "auto_insert_metric_name": False,
                    },
                }
            ],
        },
        "model": {
            "class_path": (
                "trainers.ssl4eo_mae_vit_unet_slr_task."
                "SSL4EOMAEViTUNetSLRTask"
            ),
            "init_args": {
                "checkpoint_path": str(mae_checkpoint.resolve()),
                "image_size": 256,
                "selected_blocks": [1, 4, 7, 9, 11],
                "slr_rank": 16,
                "slr_blocks": None,
                "patch_embed_adapter": True,
                "norm_trainable": True,
                "train_cls_token": True,
                "slr_checkpoint_path": str(
                    slr_checkpoint.resolve()
                ),
                "strict_slr_loading": True,
                "in_channels": 13,
                "num_classes": 6,
                "loss": "ce",
                "class_weights": None,
                "ignore_index": 0,
                "lr": lr,
                "head_lr": head_lr,
                "weight_decay": 0.05,
                "patience": 5,
            },
        },
        "data": data,
    }


def output_path(dataset: str, experiment: str) -> Path:
    return (
        REPO_ROOT
        / "experiments"
        / dataset
        / "generated_peft"
        / "slr_supervised"
        / f"{experiment}.yaml"
    )


def write_yaml(
    path: Path,
    config: dict[str, Any],
    overwrite: bool,
    dry_run: bool,
) -> str:
    existed = path.exists()
    if existed and not overwrite:
        return "SKIPPED"
    if dry_run:
        return "DRY-RUN"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=120,
        )
    return "OVERWRITTEN" if existed else "CREATED"


def main() -> int:
    args = parse_args()

    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0。")
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs 必须大于 0。")
    if args.lr <= 0 or args.head_lr <= 0:
        raise ValueError("lr 和 head_lr 必须大于 0。")

    mae_checkpoint = Path(args.mae_checkpoint).expanduser()
    if not mae_checkpoint.is_absolute():
        mae_checkpoint = (REPO_ROOT / mae_checkpoint).resolve()
    if not mae_checkpoint.is_file():
        raise FileNotFoundError(
            f"MAE checkpoint 不存在：{mae_checkpoint}"
        )

    explicit = parse_checkpoint_map(args.checkpoint_map)

    counts = {
        "created": 0,
        "overwritten": 0,
        "skipped": 0,
        "missing_checkpoint": 0,
        "failed": 0,
    }

    print("=" * 110)
    print("SLR 监督训练 YAML 生成")
    print("项目根目录：", REPO_ROOT)
    print("MAE checkpoint：", mae_checkpoint)
    print("数据集：", ", ".join(args.datasets))
    print("实验：", ", ".join(args.experiments))
    print("max_epochs：", args.max_epochs)
    print(
        "batch_size：",
        "保持参考 YAML" if args.batch_size is None else args.batch_size,
    )
    print("=" * 110)

    for dataset in args.datasets:
        for experiment in args.experiments:
            try:
                reference = find_reference_config(dataset, experiment)
                slr_checkpoint = (
                    explicit.get((dataset, experiment))
                    or find_slr_checkpoint(dataset, experiment)
                )

                if slr_checkpoint is None:
                    counts["missing_checkpoint"] += 1
                    print(
                        f"[NO CKPT    ] {dataset:10s} "
                        f"{experiment:15s} 跳过"
                    )
                    continue

                config = build_config(
                    dataset=dataset,
                    experiment=experiment,
                    reference_path=reference,
                    mae_checkpoint=mae_checkpoint,
                    slr_checkpoint=slr_checkpoint,
                    batch_size=args.batch_size,
                    max_epochs=args.max_epochs,
                    lr=args.lr,
                    head_lr=args.head_lr,
                )

                target = output_path(dataset, experiment)
                status = write_yaml(
                    path=target,
                    config=config,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )

                print(
                    f"[{status:11s}] {dataset:10s} "
                    f"{experiment:15s}\n"
                    f"  reference: {reference.relative_to(REPO_ROOT)}\n"
                    f"  slr_ckpt : {slr_checkpoint}\n"
                    f"  output   : {target.relative_to(REPO_ROOT)}"
                )

                if status == "CREATED":
                    counts["created"] += 1
                elif status == "OVERWRITTEN":
                    counts["overwritten"] += 1
                elif status == "SKIPPED":
                    counts["skipped"] += 1

            except Exception as exc:  # noqa: BLE001
                counts["failed"] += 1
                print(
                    f"[FAILED     ] {dataset:10s} "
                    f"{experiment:15s} "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    print("=" * 110)
    print(
        ", ".join(
            f"{key}={value}"
            for key, value in counts.items()
        )
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())