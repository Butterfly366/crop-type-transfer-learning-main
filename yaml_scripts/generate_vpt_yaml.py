"""
仅生成 VPT 的 4000OOD 与 4000OOD-900ID 训练/测试配置。

输入模板
--------
使用各数据集现有 FrozenViT 配置作为模板：

1. 10ID
2. 100ID
3. 900ID
4. 4000OOD
5. 4000OOD-10ID
6. 4000OOD-100ID
7. 4000OOD-900ID

对应原模板后缀分别为：

- 10ID
- 100ID
- 900ID
- 4000OOD
- 4000_10
- 4000_100
- 4000_900

输出
----
在以下目录生成训练和测试配置：

experiments/<dataset>/generated_peft/vpt/

每个数据集生成：

- 10ID.yaml
- 10ID_test.yaml
- 100ID.yaml
- 100ID_test.yaml
- 900ID.yaml
- 900ID_test.yaml
- 4000OOD.yaml
- 4000OOD_test.yaml
- 4000OOD-10ID.yaml
- 4000OOD-10ID_test.yaml
- 4000OOD-100ID.yaml
- 4000OOD-100ID_test.yaml
- 4000OOD-900ID.yaml
- 4000OOD-900ID_test.yaml

修改内容
--------
- 模型类替换为 VPT Task；
- 删除 FrozenViT 专用参数；
- 加入 VPT-Deep 参数；
- prompt_length 设为 128；
- num_workers 全部改为 12；
- 修改 default_root_dir；
- 修改 WandB project/name/save_dir；
- 修改所有 callback.dirpath（若模板中存在）；
- data、路径、split、batch_size、类别数、损失函数等保持不变。

运行
----
在项目根目录执行：

    python yaml_scripts/generate_all_vpt_configs.py
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ============================================================
# 1. 路径与实验设置
# ============================================================

# 脚本位于：
#   <repo>/yaml_scripts/generate_vpt_4000_and_4000_900.py
# 因此 parents[1] 是项目根目录。
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPO_ROOT / "experiments"

DATASETS = ("cdl", "eurocrops", "nccm", "sact", "sas")

# 原模板后缀 -> 标准输出名称。
EXPERIMENTS = {
    "10ID": "10ID",
    "100ID": "100ID",
    "900ID": "900ID",
    "4000OOD": "4000OOD",
    "4000_10": "4000OOD-10ID",
    "4000_100": "4000OOD-100ID",
    "4000_900": "4000OOD-900ID",
}

VPT_CLASS_PATH = (
    "trainers.ssl4eo_moco_vit_vpt_task."
    "SSL4EOMoCoViTUNetVPTTask"
)

# 参考 CrossEarth-Gate 的 VPT-Deep 设置。
VPT_MODEL_ARGS: dict[str, Any] = {
    "prompt_length": 128,
    "vpt_type": "deep",
    "prompt_dropout": 0.0,
}


# ============================================================
# 2. YAML 读写
# ============================================================

def read_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML，并检查顶层结构。"""
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"YAML 顶层不是字典：{path}")

    return config


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    """保存 YAML，并创建父目录。"""
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


# ============================================================
# 3. 模板路径
# ============================================================

def build_template_path(
    dataset: str,
    template_experiment: str,
    is_test: bool,
) -> Path:
    """
    构造 FrozenViT 模板路径。

    示例：
      sentinel2_cdl_ssl4eo_moco_vits16_frozen_vit_4000OOD.yaml
      sentinel2_cdl_ssl4eo_moco_vits16_frozen_vit_4000_900_test.yaml
    """
    test_suffix = "_test" if is_test else ""

    filename = (
        f"sentinel2_{dataset}_ssl4eo_moco_vits16_"
        f"frozen_vit_{template_experiment}{test_suffix}.yaml"
    )

    return (
        EXPERIMENT_ROOT
        / dataset
        / "PEFT"
        / filename
    )


def build_output_path(
    dataset: str,
    output_experiment: str,
    is_test: bool,
) -> Path:
    """构造 VPT 输出配置路径。"""
    test_suffix = "_test" if is_test else ""

    return (
        EXPERIMENT_ROOT
        / dataset
        / "generated_peft"
        / "vpt"
        / f"{output_experiment}{test_suffix}.yaml"
    )


# ============================================================
# 4. 递归修改工具
# ============================================================

def set_all_num_workers(
    value: Any,
    num_workers: int = 12,
) -> None:
    """递归将所有 num_workers 改为 12。"""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "num_workers":
                value[key] = num_workers
            else:
                set_all_num_workers(item, num_workers)

    elif isinstance(value, list):
        for item in value:
            set_all_num_workers(item, num_workers)


def set_all_dirpath(
    value: Any,
    target_dir: str,
) -> None:
    """递归修改所有 callback 中的 dirpath。"""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "dirpath" and isinstance(item, str):
                value[key] = target_dir
            else:
                set_all_dirpath(item, target_dir)

    elif isinstance(value, list):
        for item in value:
            set_all_dirpath(item, target_dir)


# ============================================================
# 5. 模型与输出设置
# ============================================================

def remove_incompatible_model_args(
    init_args: dict[str, Any],
) -> None:
    """
    删除 FrozenViT、LoRA、AdaptFormer 或旧 VPT 参数。

    这样即使模板后来被修改，也不会出现不同 PEFT 参数混杂。
    """
    keys = (
        # FrozenViT
        "freeze_vit",
        "selected_blocks",

        # LoRA
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_blocks",
        "lora_query",
        "lora_value",

        # AdaptFormer
        "adapter_bottleneck_dim",
        "adapter_scale",
        "adapter_dropout",
        "adapter_learnable_scale",
        "adapter_blocks",

        # VPT
        "prompt_length",
        "vpt_type",
        "prompt_dropout",
    )

    for key in keys:
        init_args.pop(key, None)


def update_model(config: dict[str, Any]) -> None:
    """把模板模型替换为 VPT 模型。"""
    model = config.get("model")

    if not isinstance(model, dict):
        raise KeyError("配置中不存在 model 字典。")

    init_args = model.get("init_args")

    if not isinstance(init_args, dict):
        raise KeyError("配置中不存在 model.init_args 字典。")

    remove_incompatible_model_args(init_args)

    model["class_path"] = VPT_CLASS_PATH
    init_args.update(copy.deepcopy(VPT_MODEL_ARGS))


def update_trainer_and_logger(
    config: dict[str, Any],
    dataset: str,
    experiment: str,
    is_test: bool,
) -> None:
    """更新输出目录和 WandB 名称。"""
    trainer = config.get("trainer")

    if not isinstance(trainer, dict):
        raise KeyError("配置中不存在 trainer 字典。")

    target_root = (
        f"./outputs/{dataset}/PEFT/vpt/{experiment}"
    )

    # 训练和测试配置使用相同实验根目录。
    trainer["default_root_dir"] = target_root

    # 若模板 callback 显式设置了 dirpath，则同步修改。
    set_all_dirpath(
        trainer.get("callbacks"),
        target_root,
    )

    logger = trainer.get("logger")

    if not isinstance(logger, dict):
        return

    logger_args = logger.get("init_args")

    if not isinstance(logger_args, dict):
        return

    stage = "Test" if is_test else "Train"
    dataset_upper = dataset.upper()

    logger_args["project"] = (
        f"{dataset_upper}-{experiment}-VPT"
    )
    logger_args["name"] = (
        f"{dataset_upper}-{experiment}-"
        f"SSL4EO-MoCo-ViTS16-"
        f"VPT-Deep-p128-{stage}"
    )
    logger_args["save_dir"] = (
        f"./outputs/{dataset}/PEFT/vpt"
    )


def build_vpt_config(
    source: dict[str, Any],
    dataset: str,
    experiment: str,
    is_test: bool,
) -> dict[str, Any]:
    """由 FrozenViT 模板生成 VPT 配置。"""
    config = copy.deepcopy(source)

    update_model(config)

    update_trainer_and_logger(
        config=config,
        dataset=dataset,
        experiment=experiment,
        is_test=is_test,
    )

    set_all_num_workers(
        config,
        num_workers=12,
    )

    return config


# ============================================================
# 6. 检查
# ============================================================

def validate_generated_config(
    config: dict[str, Any],
    source_config: dict[str, Any],
    output_path: Path,
) -> None:
    """对生成结果做关键一致性检查。"""
    model = config.get("model", {})
    init_args = model.get("init_args", {})

    if model.get("class_path") != VPT_CLASS_PATH:
        raise RuntimeError(
            f"VPT class_path 错误：{output_path}"
        )

    for key, expected in VPT_MODEL_ARGS.items():
        if init_args.get(key) != expected:
            raise RuntimeError(
                f"{key} 不正确：{output_path}"
            )

    for forbidden in ("freeze_vit", "selected_blocks"):
        if forbidden in init_args:
            raise RuntimeError(
                f"仍残留 {forbidden}：{output_path}"
            )

    # data 必须与模板完全一致，除 num_workers 外。
    source_data = copy.deepcopy(source_config.get("data"))
    generated_data = copy.deepcopy(config.get("data"))

    set_all_num_workers(source_data, 12)

    if generated_data != source_data:
        raise RuntimeError(
            "生成配置的 data 部分除 num_workers 外发生变化："
            f"{output_path}"
        )

    worker_values: list[Any] = []

    def collect_workers(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "num_workers":
                    worker_values.append(item)
                else:
                    collect_workers(item)
        elif isinstance(value, list):
            for item in value:
                collect_workers(item)

    collect_workers(config)

    if not worker_values or any(
        worker != 12
        for worker in worker_values
    ):
        raise RuntimeError(
            f"num_workers 未全部设为 12："
            f"{output_path} -> {worker_values}"
        )


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    if not EXPERIMENT_ROOT.is_dir():
        raise FileNotFoundError(
            f"找不到 experiments 目录：{EXPERIMENT_ROOT}"
        )

    generated = 0
    missing: list[Path] = []

    for dataset in DATASETS:
        for template_name, output_name in EXPERIMENTS.items():
            for is_test in (False, True):
                source_path = build_template_path(
                    dataset=dataset,
                    template_experiment=template_name,
                    is_test=is_test,
                )

                if not source_path.is_file():
                    missing.append(source_path)
                    continue

                output_path = build_output_path(
                    dataset=dataset,
                    output_experiment=output_name,
                    is_test=is_test,
                )

                source_config = read_yaml(source_path)

                generated_config = build_vpt_config(
                    source=source_config,
                    dataset=dataset,
                    experiment=output_name,
                    is_test=is_test,
                )

                validate_generated_config(
                    config=generated_config,
                    source_config=source_config,
                    output_path=output_path,
                )

                write_yaml(
                    path=output_path,
                    config=generated_config,
                )

                stage = "TEST" if is_test else "TRAIN"

                print(
                    f"[{stage} OK] "
                    f"{source_path.relative_to(REPO_ROOT)}"
                    f" -> "
                    f"{output_path.relative_to(REPO_ROOT)}"
                )

                generated += 1

    print("=" * 80)
    print(f"生成配置数量：{generated}")
    print(
        "预期数量："
        f"{len(DATASETS) * len(EXPERIMENTS) * 2}"
    )

    if missing:
        print("=" * 80)
        print("以下模板不存在：")

        for path in missing:
            print(
                f"  - {path.relative_to(REPO_ROOT)}"
            )

        raise FileNotFoundError(
            f"缺少 {len(missing)} 个模板，"
            "未能完整生成全部配置。"
        )

    expected = (
        len(DATASETS)
        * len(EXPERIMENTS)
        * 2
    )

    if generated != expected:
        raise RuntimeError(
            f"生成数量错误：期望 {expected}，"
            f"实际 {generated}。"
        )

    print("VPT CONFIG GENERATION PASS")


if __name__ == "__main__":
    main()