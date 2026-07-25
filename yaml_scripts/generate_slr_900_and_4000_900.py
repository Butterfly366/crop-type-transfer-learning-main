#!/usr/bin/env python3
"""
为五个数据集生成 SLR 的两阶段配置：

1. 900ID
2. 4000OOD-900ID

每个实验生成：
- SLR-MAE 自监督训练配置；
- SLR 监督分割训练配置；
- SLR 监督分割测试配置。

数据部分直接复制现有 LoRA 配置，因此 split、影像路径、标签路径、
batch size 和类别设置保持一致。

WandB project 与原有方法统一：
    <DATASET>-<EXPERIMENT>-PEFT

注意：
监督配置中的 slr_checkpoint_path 初始写为占位符：
    __SLR_MAE_CHECKPOINT__

完成自监督训练后，必须替换为对应 best .ckpt 的实际路径。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments"

DATASETS = ("cdl", "eurocrops", "nccm", "sact", "sas")
EXPERIMENTS = ("900ID", "4000OOD-900ID")

SOURCE_FILENAMES = {
    "900ID": "900ID.yaml",
    "4000OOD-900ID": "4000OOD-900ID.yaml",
}

SSL_TASK = (
    "trainers.ssl4eo_moco_vit_slr_mae_task."
    "SSL4EOMoCoViTSLRMAETask"
)
SUPERVISED_TASK = (
    "trainers.ssl4eo_moco_vit_slr_task."
    "SSL4EOMoCoViTUNetSLRTask"
)

SLR_PLACEHOLDER = "__SLR_MAE_CHECKPOINT__"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        raise TypeError(f"YAML 顶层不是字典：{path}")

    return value


def write_yaml(
    path: Path,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        yaml.safe_dump(
            config,
            handle,
            allow_unicode=True,
            sort_keys=False,
            width=1000,
        )


def source_path(
    dataset: str,
    experiment: str,
    is_test: bool = False,
) -> Path:
    suffix = "_test.yaml" if is_test else ".yaml"

    return (
        EXPERIMENT_ROOT
        / dataset
        / "generated_peft"
        / "lora"
        / f"{experiment}{suffix}"
    )


def project_name(
    dataset: str,
    experiment: str,
) -> str:
    return f"{dataset.upper()}-{experiment}-PEFT"


def update_common_trainer(
    config: dict[str, Any],
    dataset: str,
    experiment: str,
    stage_dir: str,
    run_name: str,
) -> None:
    trainer = config["trainer"]
    output_root = (
        f"./outputs/{dataset}/PEFT/slr/"
        f"{experiment}/{stage_dir}"
    )

    trainer["default_root_dir"] = output_root

    logger_args = trainer["logger"]["init_args"]
    logger_args["project"] = project_name(
        dataset,
        experiment,
    )
    logger_args["name"] = run_name
    logger_args["save_dir"] = (
        f"./outputs/{dataset}/PEFT/slr"
    )


def make_ssl_config(
    source: dict[str, Any],
    dataset: str,
    experiment: str,
) -> dict[str, Any]:
    config = copy.deepcopy(source)

    update_common_trainer(
        config,
        dataset,
        experiment,
        "ssl_pretrain",
        (
            f"{dataset.upper()}-{experiment}-"
            "SSL4EO-MoCo-ViTS16-SLR-MAE-r16-Train"
        ),
    )

    trainer = config["trainer"]
    trainer["max_epochs"] = 100

    # MAE 以 val_loss 保存最佳模型。
    trainer["callbacks"] = [
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
                    "val_loss{val_loss:.6f}"
                ),
            },
        }
    ]

    checkpoint_path = (
        source["model"]["init_args"]["checkpoint_path"]
    )

    config["model"] = {
        "class_path": SSL_TASK,
        "init_args": {
            "checkpoint_path": checkpoint_path,
            "image_size": 256,
            "in_channels": 13,
            "patch_size": 16,
            "mask_ratio": 0.75,
            "slr_rank": 16,
            "slr_blocks": None,
            "decoder_embed_dim": 512,
            "decoder_depth": 8,
            "decoder_num_heads": 16,
            "norm_pix_loss": False,
            "lr": 1.0e-4,
            "weight_decay": 0.05,
            "min_lr": 0.0,
            "warmup_epochs": 10,
        },
    }

    return config


def make_supervised_config(
    source: dict[str, Any],
    dataset: str,
    experiment: str,
    is_test: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(source)
    stage = "Test" if is_test else "Train"

    update_common_trainer(
        config,
        dataset,
        experiment,
        "supervised",
        (
            f"{dataset.upper()}-{experiment}-"
            f"SSL4EO-MoCo-ViTS16-SLR-r16-{stage}"
        ),
    )

    source_args = source["model"]["init_args"]

    config["model"] = {
        "class_path": SUPERVISED_TASK,
        "init_args": {
            "checkpoint_path": source_args["checkpoint_path"],
            "image_size": 256,
            "slr_rank": 16,
            "slr_blocks": None,
            "slr_qkv": True,
            "slr_attn_proj": True,
            "slr_mlp_fc1": True,
            "slr_mlp_fc2": True,
            "slr_checkpoint_path": SLR_PLACEHOLDER,
            "strict_slr_loading": True,
            "in_channels": source_args.get(
                "in_channels",
                13,
            ),
            "num_classes": source_args.get(
                "num_classes",
                6,
            ),
            "loss": source_args.get("loss", "ce"),
            "ignore_index": source_args.get(
                "ignore_index",
                0,
            ),
            "lr": source_args.get("lr", 1.0e-3),
            "patience": source_args.get(
                "patience",
                1,
            ),
        },
    }

    return config


def main() -> None:
    generated = 0
    missing: list[Path] = []

    for dataset in DATASETS:
        for experiment in EXPERIMENTS:
            train_source_path = source_path(
                dataset,
                experiment,
                is_test=False,
            )
            test_source_path = source_path(
                dataset,
                experiment,
                is_test=True,
            )

            if not train_source_path.is_file():
                missing.append(train_source_path)
                continue

            if not test_source_path.is_file():
                missing.append(test_source_path)
                continue

            train_source = read_yaml(
                train_source_path
            )
            test_source = read_yaml(
                test_source_path
            )

            base = (
                EXPERIMENT_ROOT
                / dataset
                / "generated_peft"
                / "slr"
            )

            outputs = (
                (
                    base / "ssl" / f"{experiment}.yaml",
                    make_ssl_config(
                        train_source,
                        dataset,
                        experiment,
                    ),
                ),
                (
                    base
                    / "supervised"
                    / f"{experiment}.yaml",
                    make_supervised_config(
                        train_source,
                        dataset,
                        experiment,
                        is_test=False,
                    ),
                ),
                (
                    base
                    / "supervised"
                    / f"{experiment}_test.yaml",
                    make_supervised_config(
                        test_source,
                        dataset,
                        experiment,
                        is_test=True,
                    ),
                ),
            )

            for output_path, config in outputs:
                write_yaml(output_path, config)
                print(
                    "[OK]",
                    output_path.relative_to(REPO_ROOT),
                )
                generated += 1

    if missing:
        print("缺少以下 LoRA 模板：")

        for path in missing:
            print(" -", path.relative_to(REPO_ROOT))

        raise FileNotFoundError(
            f"缺少 {len(missing)} 个模板。"
        )

    expected = (
        len(DATASETS)
        * len(EXPERIMENTS)
        * 3
    )

    if generated != expected:
        raise RuntimeError(
            f"生成数量错误：{generated}/{expected}"
        )

    print(f"生成配置数量：{generated}")
    print("SLR STEP 5 CONFIG GENERATION PASS")


if __name__ == "__main__":
    main()
