"""批量生成 SSL4EO MAE ViT-S/16 + SLR + UNet 最终测试 YAML。"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("cdl", "eurocrops", "nccm", "sact", "sas")
EXPERIMENTS = ("900ID", "4000OOD-900ID")
DISPLAY = {
    "cdl": "CDL",
    "eurocrops": "EuroCrops",
    "nccm": "NCCM",
    "sact": "SACT",
    "sas": "SAS",
}
TEST_TEMPLATES = (
    "experiments/{dataset}/generated_peft/lora/{experiment}_test.yaml",
    "experiments/{dataset}/generated_peft/adaptformer/{experiment}_test.yaml",
    "experiments/{dataset}/PEFT/lora/{experiment}_test.yaml",
    "experiments/{dataset}/PEFT/adaptformer/{experiment}_test.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML 根节点不是映射：{path}")
    return value


def train_path(dataset: str, experiment: str) -> Path:
    return (
        REPO
        / "experiments"
        / dataset
        / "generated_peft"
        / "slr_supervised"
        / f"{experiment}.yaml"
    )


def output_path(dataset: str, experiment: str) -> Path:
    return train_path(dataset, experiment).with_name(f"{experiment}_test.yaml")


def find_test_template(dataset: str, experiment: str) -> Path:
    checked: list[Path] = []
    for pattern in TEST_TEMPLATES:
        candidate = REPO / pattern.format(dataset=dataset, experiment=experiment)
        checked.append(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"找不到 {dataset}/{experiment} 的测试数据模板：\n"
        + "\n".join(f"  - {path}" for path in checked)
    )


def build_logger(train: dict[str, Any], dataset: str, experiment: str) -> dict[str, Any]:
    logger = copy.deepcopy(train.get("trainer", {}).get("logger", {}))
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

    name = DISPLAY[dataset]
    init_args["project"] = f"{name}-{experiment}-SLR-Supervised"
    init_args["name"] = (
        f"{name}-{experiment}-SSL4EO-MAE-ViTS16-SLR-r16-UNet-Test"
    )
    init_args["save_dir"] = f"./outputs/{dataset}/PEFT/slr_supervised"
    init_args.setdefault("offline", False)
    init_args.setdefault("log_model", False)
    return logger


def build_config(
    dataset: str,
    experiment: str,
    supervised_train: Path,
    test_template: Path,
    batch_size: int | None,
) -> dict[str, Any]:
    train = load_yaml(supervised_train)
    template = load_yaml(test_template)

    model = copy.deepcopy(train.get("model"))
    data = copy.deepcopy(template.get("data"))
    if not isinstance(model, dict):
        raise ValueError(f"监督训练 YAML 缺少 model：{supervised_train}")
    if not isinstance(data, dict):
        raise ValueError(f"测试模板缺少 data：{test_template}")

    data_args = data.setdefault("init_args", {})
    if not isinstance(data_args, dict):
        raise ValueError("data.init_args 必须是字典。")
    if batch_size is not None:
        data_args["batch_size"] = batch_size

    config = {
        "seed_everything": train.get("seed_everything", 0),
        "trainer": {
            "accelerator": "gpu",
            "devices": 1,
            "logger": build_logger(train, dataset, experiment),
            "log_every_n_steps": 1,
            "enable_checkpointing": False,
            "default_root_dir": (
                f"./outputs/{dataset}/PEFT/slr_supervised/{experiment}/test"
            ),
        },
        "model": model,
        "data": data,
    }

    expected = (
        "trainers.ssl4eo_mae_vit_unet_slr_task."
        "SSL4EOMAEViTUNetSLRTask"
    )
    if model.get("class_path") != expected:
        raise ValueError(f"监督训练 YAML 使用了错误 Task：{model.get('class_path')}")
    if train.get("model") != config.get("model"):
        raise ValueError("训练与测试 model 配置不完全一致。")
    if int(data_args.get("batch_size", 0)) <= 0:
        raise ValueError("测试 batch_size 无效。")

    return config


def write_yaml(path: Path, config: dict[str, Any], overwrite: bool, dry_run: bool) -> str:
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

    failed = 0
    print("=" * 100)
    print("SLR 最终测试 YAML 生成")
    print("项目根目录：", REPO)
    print("=" * 100)

    for dataset in args.datasets:
        for experiment in args.experiments:
            supervised_train = train_path(dataset, experiment)
            if not supervised_train.is_file():
                print(f"[NO TRAIN] {dataset:10s} {experiment:15s} 缺少监督训练 YAML")
                continue

            try:
                template = find_test_template(dataset, experiment)
                config = build_config(
                    dataset,
                    experiment,
                    supervised_train,
                    template,
                    args.batch_size,
                )
                target = output_path(dataset, experiment)
                status = write_yaml(target, config, args.overwrite, args.dry_run)
                print(
                    f"[{status:11s}] {dataset:10s} {experiment:15s}\n"
                    f"  train    : {supervised_train.relative_to(REPO)}\n"
                    f"  test data: {template.relative_to(REPO)}\n"
                    f"  output   : {target.relative_to(REPO)}"
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(
                    f"[FAILED] {dataset:10s} {experiment:15s} "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())