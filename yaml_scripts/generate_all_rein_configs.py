#!/usr/bin/env python3
"""
为五个数据集生成 REIN 训练和测试配置。

生成范围：
- 10ID
- 100ID
- 900ID
- 4000OOD
- 4000OOD-10ID
- 4000OOD-100ID
- 4000OOD-900ID

每个实验生成：
- train YAML
- test YAML

数据和训练部分直接复制现有 LoRA 配置，只替换：
- model.class_path
- REIN 模型参数
- logger name
- save_dir
- default_root_dir
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments"

DATASETS = (
    "cdl",
    "eurocrops",
    "nccm",
    "sact",
    "sas",
)

EXPERIMENTS = (
    "10ID",
    "100ID",
    "900ID",
    "4000OOD",
    "4000OOD-10ID",
    "4000OOD-100ID",
    "4000OOD-900ID",
)

TASK_CLASS = (
    "trainers.ssl4eo_moco_vit_rein_task."
    "SSL4EOMoCoViTUNetREINTask"
)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(
        encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError(
            f"YAML 顶层不是字典：{path}"
        )

    return config


def write_yaml(
    path: Path,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
    is_test: bool,
) -> Path:
    suffix = "_test.yaml" if is_test else ".yaml"

    return (
        EXPERIMENT_ROOT
        / dataset
        / "generated_peft"
        / "lora"
        / f"{experiment}{suffix}"
    )


def make_rein_config(
    source: dict[str, Any],
    dataset: str,
    experiment: str,
    is_test: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(source)
    stage = "Test" if is_test else "Train"

    output_root = (
        f"./outputs/{dataset}/PEFT/rein/"
        f"{experiment}"
    )

    trainer = config["trainer"]
    trainer["default_root_dir"] = output_root

    logger_args = trainer["logger"]["init_args"]
    logger_args["project"] = (
        f"{dataset.upper()}-{experiment}-PEFT"
    )
    logger_args["name"] = (
        f"{dataset.upper()}-{experiment}-"
        f"SSL4EO-MoCo-ViTS16-"
        f"LoRA-REIN-t100-r16-{stage}"
    )
    logger_args["save_dir"] = (
        f"./outputs/{dataset}/PEFT/rein"
    )

    source_args = source["model"]["init_args"]

    config["model"] = {
        "class_path": TASK_CLASS,
        "init_args": {
            "checkpoint_path": (
                source_args["checkpoint_path"]
            ),
            "image_size": 256,
            "rein_type": "lora_rein",
            "token_length": 100,
            "lora_dim": 16,
            "use_softmax": True,
            "scale_init": 0.001,
            "selected_blocks": [
                1,
                4,
                7,
                9,
                11,
            ],
            "in_channels": source_args.get(
                "in_channels",
                13,
            ),
            "num_classes": source_args.get(
                "num_classes",
                6,
            ),
            "loss": source_args.get(
                "loss",
                "ce",
            ),
            "ignore_index": source_args.get(
                "ignore_index",
                0,
            ),
            "lr": source_args.get(
                "lr",
                1.0e-3,
            ),
            "patience": source_args.get(
                "patience",
                1,
            ),
        },
    }

    # 统一新配置的数据加载线程数。
    config["data"]["init_args"]["num_workers"] = 12

    return config


def main() -> None:
    generated = 0
    missing: list[Path] = []

    for dataset in DATASETS:
        for experiment in EXPERIMENTS:
            for is_test in (False, True):
                source = source_path(
                    dataset,
                    experiment,
                    is_test,
                )

                if not source.is_file():
                    missing.append(source)
                    continue

                config = make_rein_config(
                    read_yaml(source),
                    dataset,
                    experiment,
                    is_test,
                )

                suffix = (
                    "_test.yaml"
                    if is_test
                    else ".yaml"
                )

                output = (
                    EXPERIMENT_ROOT
                    / dataset
                    / "generated_peft"
                    / "rein"
                    / f"{experiment}{suffix}"
                )

                write_yaml(output, config)

                print(
                    "[OK]",
                    output.relative_to(REPO_ROOT),
                )
                generated += 1

    if missing:
        print(
            "\n以下 LoRA 模板不存在，"
            "对应 REIN 配置未生成："
        )

        for path in missing:
            print(
                " -",
                path.relative_to(REPO_ROOT),
            )

    expected = (
        len(DATASETS)
        * len(EXPERIMENTS)
        * 2
    )

    print(
        f"\n生成配置数量：{generated}/{expected}"
    )

    if generated == expected:
        print(
            "REIN STEP 3 CONFIG GENERATION PASS"
        )
    else:
        print(
            "部分模板缺失，请核对上方列表。"
        )


if __name__ == "__main__":
    main()
