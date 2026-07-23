"""
根据现有 FrozenViT 配置，批量生成 LoRA 和 AdaptFormer 的训练与测试配置。

目标实验：
- 数据集：cdl、eurocrops、nccm、sact、sas
- ID：10ID、100ID、900ID
- OOD：4000OOD、4000OOD-10ID、4000OOD-100ID、4000OOD-900ID

输出示例：
experiments/cdl/generated_peft/lora/100ID.yaml
experiments/cdl/generated_peft/lora/100ID_test.yaml
experiments/cdl/generated_peft/adaptformer/100ID.yaml
experiments/cdl/generated_peft/adaptformer/100ID_test.yaml
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
GENERATED_DIR_NAME = "generated_peft"


METHODS: dict[str, dict[str, Any]] = {
    "lora": {
        "class_path": (
            "trainers.ssl4eo_moco_vit_lora_task."
            "SSL4EOMoCoViTUNetLoRATask"
        ),
        "model_args": {
            "lora_rank": 4,
            "lora_alpha": 4.0,
            "lora_dropout": 0.0,
            "lora_blocks": None,
            "lora_query": True,
            "lora_value": True,
        },
        "display_name": "LoRA-r4",
    },
    "adaptformer": {
        "class_path": (
            "trainers.ssl4eo_moco_vit_adaptformer_task."
            "SSL4EOMoCoViTUNetAdaptFormerTask"
        ),
        "model_args": {
            "adapter_bottleneck_dim": 64,
            "adapter_scale": 0.1,
            "adapter_dropout": 0.0,
            "adapter_learnable_scale": False,
            "adapter_blocks": None,
        },
        "display_name": "AdaptFormer-d64",
    },
}


DATASET_NAMES = ("cdl", "eurocrops", "nccm", "sact", "sas")


# 同时兼容以下模板命名：
#   10ID / 100ID / 900ID
#   4000OOD
#   4000OOD-10ID / 4000OOD-100ID / 4000OOD-900ID
#   4000
#   4000_10 / 4000_100 / 4000_900
#
# 注意：优先匹配 4000 组合实验，避免把 4000_100 错识别成 100ID。
EXPERIMENT_PATTERN = re.compile(
    r"(^|[^0-9])("
    r"4000OOD-(?:10|100|900)ID|"
    r"4000OOD|"
    r"4000_(?:10|100|900)|"
    r"4000|"
    r"10ID|100ID|900ID"
    r")([^0-9]|$)",
    flags=re.IGNORECASE,
)


def normalize_experiment_name(raw_name: str) -> str:
    """
    将不同模板命名统一为标准输出名称。

    示例：
        4000          -> 4000OOD
        4000_10       -> 4000OOD-10ID
        4000_100      -> 4000OOD-100ID
        4000_900      -> 4000OOD-900ID
        100ID         -> 100ID
    """
    name = raw_name.upper()

    alias_map = {
        "4000": "4000OOD",
        "4000_10": "4000OOD-10ID",
        "4000_100": "4000OOD-100ID",
        "4000_900": "4000OOD-900ID",
    }

    return alias_map.get(name, name)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"YAML 顶层不是字典：{path}")

    return config


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(
            config,
            file,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            width=1000,
        )


def find_model_section(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model")

    if not isinstance(model, dict):
        raise KeyError("配置中不存在字典形式的 model。")

    init_args = model.setdefault("init_args", {})

    if not isinstance(init_args, dict):
        raise TypeError("model.init_args 必须是字典。")

    return model


def infer_dataset(path: Path) -> str:
    lower_path = path.as_posix().lower()

    for dataset in DATASET_NAMES:
        if dataset in lower_path:
            return dataset

    raise ValueError(f"无法识别数据集：{path}")


def infer_experiment_name(path: Path) -> str:
    """
    从配置路径中提取实验名称，并统一成标准命名。
    """
    match = EXPERIMENT_PATTERN.search(path.as_posix())

    if match is None:
        raise ValueError(f"无法识别实验名称：{path}")

    return normalize_experiment_name(match.group(2))


def is_frozen_vit_train_template(
    path: Path,
    config: dict[str, Any],
) -> bool:
    if path.stem.lower().endswith("_test"):
        return False

    lower_path = path.as_posix().lower()

    if not any(dataset in lower_path for dataset in DATASET_NAMES):
        return False

    if EXPERIMENT_PATTERN.search(path.as_posix()) is None:
        return False

    model = config.get("model")
    if not isinstance(model, dict):
        return False

    class_path = str(model.get("class_path", ""))
    init_args = model.get("init_args", {})

    if not isinstance(init_args, dict):
        return False

    is_ssl4eo_task = (
        "SSL4EOMoCoViTUNetTask" in class_path
        or "ssl4eo_moco_vit_task" in class_path
    )

    return is_ssl4eo_task and init_args.get("freeze_vit") is True


def find_test_template(train_path: Path) -> Path | None:
    candidates = [
        train_path.with_name(
            f"{train_path.stem}_test{train_path.suffix}"
        ),
        train_path.with_name(
            f"{train_path.stem}_test.yaml"
        ),
        train_path.with_name(
            f"{train_path.stem}_test.yml"
        ),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def remove_old_peft_args(init_args: dict[str, Any]) -> None:
    keys_to_remove = (
        "freeze_vit",
        "selected_blocks",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_blocks",
        "lora_query",
        "lora_value",
        "adapter_bottleneck_dim",
        "adapter_scale",
        "adapter_dropout",
        "adapter_learnable_scale",
        "adapter_blocks",
    )

    for key in keys_to_remove:
        init_args.pop(key, None)


def update_output_dirs(
    config: dict[str, Any],
    dataset: str,
    method: str,
    experiment_name: str,
) -> dict[str, Any]:
    """
    更新当前实验的输出目录。

    同时修改：
    1. trainer.default_root_dir
    2. 所有 callback 中的 dirpath

    训练配置与测试配置共用同一个实验根目录，例如：

        ./outputs/cdl/PEFT/lora/100ID
    """
    result = copy.deepcopy(config)

    target_dir = (
        f"./outputs/{dataset}/PEFT/{method}/{experiment_name}"
    )

    trainer = result.get("trainer")

    if isinstance(trainer, dict):
        trainer["default_root_dir"] = target_dir

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "dirpath" and isinstance(item, str):
                    value[key] = target_dir
                else:
                    visit(item)

        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(result)
    return result


def update_logger(
    config: dict[str, Any],
    dataset: str,
    experiment_name: str,
    method: str,
    is_test: bool,
) -> None:
    trainer = config.get("trainer")
    if not isinstance(trainer, dict):
        return

    logger = trainer.get("logger")
    if not isinstance(logger, dict):
        return

    logger_args = logger.get("init_args")
    if not isinstance(logger_args, dict):
        return

    display_name = METHODS[method]["display_name"]
    stage_name = "Test" if is_test else "Train"

    logger_args["project"] = (
        f"{dataset.upper()}-{experiment_name}-PEFT"
    )
    logger_args["name"] = (
        f"{dataset.upper()}-{experiment_name}-"
        f"SSL4EO-MoCo-ViTS16-{display_name}-{stage_name}"
    )
    logger_args["save_dir"] = (
        f"./outputs/{dataset}/PEFT/{method}"
    )


def set_all_num_workers(
    config: dict[str, Any],
    num_workers: int = 12,
) -> None:
    """
    递归将配置中所有名为 num_workers 的字段统一改为指定值。

    这样可以同时覆盖：
    - data.init_args.num_workers
    - 其他嵌套数据模块中的 num_workers
    """
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "num_workers":
                    value[key] = num_workers
                else:
                    visit(item)

        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(config)


def build_peft_config(
    source_config: dict[str, Any],
    method: str,
    dataset: str,
    experiment_name: str,
    is_test: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(source_config)

    model = find_model_section(config)
    init_args = model["init_args"]

    remove_old_peft_args(init_args)

    model["class_path"] = METHODS[method]["class_path"]
    init_args.update(copy.deepcopy(METHODS[method]["model_args"]))

    update_logger(
        config=config,
        dataset=dataset,
        experiment_name=experiment_name,
        method=method,
        is_test=is_test,
    )

    # 所有训练与测试配置中的 num_workers 统一设为 12。
    set_all_num_workers(
        config=config,
        num_workers=12,
    )

    # 训练和测试配置都更新 default_root_dir 与 callback dirpath。
    # 同一实验的训练与测试共用一个输出根目录。
    config = update_output_dirs(
        config=config,
        dataset=dataset,
        method=method,
        experiment_name=experiment_name,
    )

    return config


def output_filename(
    source_path: Path,
    experiment_name: str,
    is_test: bool,
) -> str:
    suffix = source_path.suffix.lower()

    if suffix not in (".yaml", ".yml"):
        suffix = ".yaml"

    test_suffix = "_test" if is_test else ""

    return f"{experiment_name}{test_suffix}{suffix}"


def build_target_path(
    dataset: str,
    method: str,
    experiment_name: str,
    source_path: Path,
    is_test: bool,
) -> Path:
    return (
        EXPERIMENT_ROOT
        / dataset
        / GENERATED_DIR_NAME
        / method
        / output_filename(
            source_path=source_path,
            experiment_name=experiment_name,
            is_test=is_test,
        )
    )


def main() -> None:
    if not EXPERIMENT_ROOT.exists():
        raise FileNotFoundError(
            f"找不到 experiments 目录：{EXPERIMENT_ROOT}"
        )

    yaml_paths = sorted(
        list(EXPERIMENT_ROOT.rglob("*.yaml"))
        + list(EXPERIMENT_ROOT.rglob("*.yml"))
    )

    train_templates: list[tuple[Path, dict[str, Any]]] = []

    for path in yaml_paths:
        if GENERATED_DIR_NAME in path.parts:
            continue

        if path.stem.lower().endswith("_test"):
            continue

        try:
            config = read_yaml(path)
        except Exception as error:
            print(f"[SKIP] 无法读取 {path}: {error}")
            continue

        if is_frozen_vit_train_template(path, config):
            train_templates.append((path, config))

    if not train_templates:
        raise RuntimeError(
            "没有找到 FrozenViT 训练模板。\n"
            "请确认：\n"
            "1. model.class_path 使用 SSL4EOMoCoViTUNetTask；\n"
            "2. model.init_args.freeze_vit 为 true；\n"
            "3. 路径包含目标数据集和实验名称；\n"
            "4. 文件名不是 *_test.yaml。"
        )

    generated_train = 0
    generated_test = 0
    missing_test_templates: list[Path] = []
    seen_targets: dict[Path, Path] = {}

    for train_source_path, train_source_config in train_templates:
        dataset = infer_dataset(train_source_path)
        experiment_name = infer_experiment_name(train_source_path)

        test_source_path = find_test_template(train_source_path)

        if test_source_path is None:
            missing_test_templates.append(train_source_path)
            test_source_config = None
        else:
            try:
                test_source_config = read_yaml(test_source_path)
            except Exception as error:
                print(
                    f"[TEST SKIP] 无法读取 {test_source_path}: {error}"
                )
                missing_test_templates.append(train_source_path)
                test_source_path = None
                test_source_config = None

        for method in METHODS:
            train_target_path = build_target_path(
                dataset=dataset,
                method=method,
                experiment_name=experiment_name,
                source_path=train_source_path,
                is_test=False,
            )

            if train_target_path in seen_targets:
                previous_source = seen_targets[train_target_path]
                raise RuntimeError(
                    "多个训练模板会生成同一个目标文件：\n"
                    f"第一个模板：{previous_source}\n"
                    f"当前模板：{train_source_path}\n"
                    f"目标文件：{train_target_path}"
                )

            train_config = build_peft_config(
                source_config=train_source_config,
                method=method,
                dataset=dataset,
                experiment_name=experiment_name,
                is_test=False,
            )

            write_yaml(train_target_path, train_config)
            seen_targets[train_target_path] = train_source_path
            generated_train += 1

            print(
                "[TRAIN OK] "
                f"{train_source_path.relative_to(PROJECT_ROOT)}"
                " -> "
                f"{train_target_path.relative_to(PROJECT_ROOT)}"
            )

            if test_source_path is None or test_source_config is None:
                continue

            test_target_path = build_target_path(
                dataset=dataset,
                method=method,
                experiment_name=experiment_name,
                source_path=test_source_path,
                is_test=True,
            )

            if test_target_path in seen_targets:
                previous_source = seen_targets[test_target_path]
                raise RuntimeError(
                    "多个测试模板会生成同一个目标文件：\n"
                    f"第一个模板：{previous_source}\n"
                    f"当前模板：{test_source_path}\n"
                    f"目标文件：{test_target_path}"
                )

            test_config = build_peft_config(
                source_config=test_source_config,
                method=method,
                dataset=dataset,
                experiment_name=experiment_name,
                is_test=True,
            )

            write_yaml(test_target_path, test_config)
            seen_targets[test_target_path] = test_source_path
            generated_test += 1

            print(
                "[TEST OK] "
                f"{test_source_path.relative_to(PROJECT_ROOT)}"
                " -> "
                f"{test_target_path.relative_to(PROJECT_ROOT)}"
            )

    print("=" * 80)
    print(f"训练模板数量：{len(train_templates)}")
    print(f"生成训练配置：{generated_train}")
    print(f"生成测试配置：{generated_test}")
    print(
        "输出结构："
        "experiments/<dataset>/generated_peft/<method>/"
    )

    if missing_test_templates:
        print("=" * 80)
        print("以下训练配置没有找到对应的 *_test.yaml：")

        for path in missing_test_templates:
            print(f"  - {path.relative_to(PROJECT_ROOT)}")

    print("CONFIG GENERATION PASS")


if __name__ == "__main__":
    main()