#!/usr/bin/env python3
"""双 GPU、可断点续跑的 generated_peft 实验调度器。

仅调度五个数据集的 LoRA/AdaptFormer，实验规模严格限定为 900ID 和
4000OOD-900ID。每个 worker 在同一张 GPU 上串行执行 fit -> checkpoint ->
test；worker 失败不会阻断另一张 GPU，也不会阻断本 worker 的后续任务。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Process
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[1]
# 直接执行 scripts/ 下的文件时，Python 默认只把 scripts/ 加入 sys.path；
# 显式加入项目根目录，才能按 TorchGeo 配置导入 trainers.*。
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
LOG_DIR = REPO / "batch_logs" / "generated_peft_900_and_4000_900"
STATE_PATH = LOG_DIR / "state.jsonl"
DATASETS = ("cdl", "eurocrops", "nccm", "sact", "sas")
METHODS = ("lora", "adaptformer")
EXPERIMENTS = ("900ID", "4000OOD-900ID")
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None


@dataclass(frozen=True)
class Job:
    index: int
    dataset: str
    method: str
    experiment: str

    @property
    def train_config(self) -> Path:
        return REPO / "experiments" / self.dataset / "generated_peft" / self.method / f"{self.experiment}.yaml"

    @property
    def test_config(self) -> Path:
        return self.train_config.with_name(f"{self.experiment}_test.yaml")

    @property
    def stem(self) -> str:
        return f"{self.dataset}_{self.method}_{self.experiment}"


# 保持用户指定的顺序：先全部 900ID，再全部 4000OOD-900ID；每种规模内按
# dataset，再按 lora/adaptformer。奇数项固定 GPU 0，偶数项固定 GPU 1。
JOBS = tuple(
    Job(index, dataset, method, experiment)
    for index, (experiment, dataset, method) in enumerate(
        (experiment, dataset, method)
        for experiment in EXPERIMENTS
        for dataset in DATASETS
        for method in METHODS
    )
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML 根节点不是映射：{path}")
    return value


def get_default_root(config: dict[str, Any]) -> Path:
    raw = config.get("trainer", {}).get("default_root_dir")
    if not isinstance(raw, str) or not raw:
        raise ValueError("缺少 trainer.default_root_dir")
    return (REPO / raw).resolve() if not Path(raw).is_absolute() else Path(raw)


def get_checkpoint_root(config: dict[str, Any]) -> Path:
    """返回本仓库实际的 ModelCheckpoint 根目录。

    未显式设置 callback.dirpath 时，Lightning 会跟随 WandbLogger，将 checkpoint
    写入 logger.save_dir/project/<run>/checkpoints，而非 default_root_dir。
    """
    logger_args = config.get("trainer", {}).get("logger", {}).get("init_args", {})
    save_dir = logger_args.get("save_dir")
    project = logger_args.get("project")
    if not isinstance(save_dir, str) or not isinstance(project, str) or not save_dir or not project:
        raise ValueError("缺少 logger.save_dir 或 logger.project，无法限定 checkpoint 目录")
    base = Path(save_dir)
    return (REPO / base / project).resolve() if not base.is_absolute() else base / project


def walk_config(value: Any, key: str = "") -> list[tuple[str, Any]]:
    result = [(key, value)]
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            result.extend(walk_config(child_value, child_key))
    elif isinstance(value, list):
        for child_value in value:
            result.extend(walk_config(child_value, key))
    return result


def configured_paths(config: dict[str, Any]) -> list[tuple[str, Path]]:
    """提取配置中的本地路径；输出目录不作为输入数据路径检查。"""
    results: list[tuple[str, Path]] = []
    for key, value in walk_config(config):
        if not isinstance(value, str):
            continue
        if key == "default_root_dir" or not any(token in key.lower() for token in ("path", "root", "csv", "split")):
            continue
        if not (value.startswith("/") or value.startswith("./") or value.startswith("../")):
            continue
        path = Path(value)
        results.append((key, path if path.is_absolute() else (REPO / path).resolve()))
    return results


def preflight(job: Job) -> tuple[list[str], Path]:
    """返回所有阻塞原因及唯一的 checkpoint 输出根目录。"""
    problems: list[str] = []
    if not job.train_config.is_file():
        return ([f"训练配置缺失：{job.train_config}"], REPO)
    if not job.test_config.is_file():
        return ([f"测试配置缺失：{job.test_config}"], REPO)
    train = load_yaml(job.train_config)
    test = load_yaml(job.test_config)
    train_task = train.get("model", {}).get("class_path", "")
    test_task = test.get("model", {}).get("class_path", "")
    expected = "lora_task" if job.method == "lora" else "adaptformer_task"
    if expected not in train_task.lower() or expected not in test_task.lower():
        problems.append(f"{job.method} 配置使用了错误 Task：{train_task} / {test_task}")
    if train_task != test_task or train.get("model") != test.get("model"):
        problems.append("训练与测试配置的模型参数不一致")
    try:
        module, name = train_task.rsplit(".", 1)
        getattr(importlib.import_module(module), name)
    except Exception as exc:  # noqa: BLE001 - 需把真实导入错误写入检查表
        problems.append(f"trainer class_path 无法导入：{type(exc).__name__}: {exc}")
    worker_values = [value for key, value in walk_config(train) if key == "num_workers"]
    if not worker_values or any(value != 12 for value in worker_values):
        problems.append(f"num_workers 不是全部 12：{worker_values}")
    for key, path in configured_paths(train):
        if not path.exists():
            problems.append(f"数据/权重路径不存在：{key}={path}")
    try:
        root = get_checkpoint_root(train)
    except ValueError as exc:
        problems.append(str(exc))
        root = REPO
    return problems, root


def append_state(job: Job, gpu: int, stage: str, status: str, **extra: Any) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "dataset": job.dataset,
        "method": job.method,
        "experiment": job.experiment,
        "gpu": gpu,
        "stage": stage,
        "status": status,
        **extra,
    }
    with STATE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_path(job: Job, gpu: int, stage: str, retry: bool = False) -> Path:
    base = LOG_DIR / f"{job.stem}_gpu{gpu}_{stage}.log"
    if retry and base.exists():
        return base.with_name(f"{base.stem}_retry_{datetime.now():%Y%m%d_%H%M%S}.log")
    return base


def contains_test_success(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return "test_average_F1-score" in text and ("Synced" in text or "Testing DataLoader" in text)


def contains_fit_success(path: Path) -> bool:
    """训练完整结束的日志必须同时包含 Lightning 结束标记与 W&B 收尾标记。"""
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return "Trainer.fit` stopped" in text and "Synced" in text


def log_failed(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return bool(re.search(r"Traceback \(most recent call last\)|RuntimeError:|Error:|Exception:", text))


def checkpoint_from_log(path: Path, output_root: Path) -> Path | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"(?:best_model_path|checkpoint path|checkpoint at)[^\n]*?([/~.\w\-/=]+\.ckpt)", text, flags=re.I)
    for raw in reversed(matches):
        candidate = Path(raw).expanduser()
        candidate = candidate if candidate.is_absolute() else (REPO / candidate).resolve()
        if candidate.is_file() and output_root in candidate.parents:
            return candidate
    return None


def select_checkpoint(output_root: Path, fit_log: Path, started_at: float | None = None) -> Path | None:
    """只在当前实验 logger/project 根目录中选择 checkpoint，且遵循指定优先级。"""
    logged = checkpoint_from_log(fit_log, output_root)
    if logged:
        return logged
    if not output_root.is_dir():
        return None
    candidates = [path for path in output_root.rglob("*.ckpt") if path.is_file()]
    if started_at is not None:
        candidates = [path for path in candidates if path.stat().st_mtime >= started_at - 1]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    best_named = [path for path in candidates if "best" in path.name.lower()]
    # 本项目 callback 的 save_top_k=1 文件名含 val_f1，因此非 last 的最新文件就是最优记录。
    callback_best = [path for path in candidates if path.name != "last.ckpt"]
    return (best_named or callback_best or candidates)[0]


def run_with_tee(command: list[str], log: Path) -> int:
    """使用 bash 的 pipefail + tee，并返回 Python 训练/测试进程的真实退出码。"""
    log.parent.mkdir(parents=True, exist_ok=True)
    command_text = " ".join(shlex.quote(item) for item in command)
    shell = f"set -o pipefail; {command_text} 2>&1 | tee {shlex.quote(str(log))}; exit ${{PIPESTATUS[0]}}"
    global _ACTIVE_PROCESS
    _ACTIVE_PROCESS = subprocess.Popen(["bash", "-lc", shell], cwd=REPO, start_new_session=True)
    try:
        return _ACTIVE_PROCESS.wait()
    finally:
        _ACTIVE_PROCESS = None


def stop_active_child(_signal_number: int, _frame: Any) -> None:
    """worker 收到终止信号时杀掉自身启动的整个 shell/torchgeo 进程组。"""
    if _ACTIVE_PROCESS is not None and _ACTIVE_PROCESS.poll() is None:
        os.killpg(_ACTIVE_PROCESS.pid, signal.SIGTERM)
        try:
            _ACTIVE_PROCESS.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(_ACTIVE_PROCESS.pid, signal.SIGKILL)
    raise SystemExit(128 + _signal_number)


def run_job(job: Job, gpu: int) -> None:
    problems, output_root = preflight(job)
    if problems:
        append_state(job, gpu, "preflight", "BLOCKED", problems=problems)
        return
    default_fit = log_path(job, gpu, "fit")
    default_test = log_path(job, gpu, "test")
    existing_ckpt = select_checkpoint(output_root, default_fit)
    fit_complete = contains_fit_success(default_fit)
    if existing_ckpt and fit_complete and contains_test_success(default_test):
        append_state(job, gpu, "pipeline", "SKIPPED", checkpoint=str(existing_ckpt))
        return

    # 有 checkpoint 而无成功测试日志：直接测试；测试失败历史使用时间戳新日志。
    checkpoint = existing_ckpt if fit_complete else None
    if checkpoint is None:
        # 中断、异常或未完整结束的训练日志均保留；重跑必须写新时间戳日志。
        fit_log = log_path(job, gpu, "fit", retry=default_fit.exists())
        started_at = time.time()
        append_state(job, gpu, "fit", "RUNNING", pid=os.getpid(), log=str(fit_log))
        fit_command = [f"CUDA_VISIBLE_DEVICES={gpu}", sys.executable, "-m", "torchgeo", "fit", "--config", str(job.train_config)]
        fit_rc = run_with_tee(fit_command, fit_log)
        if fit_rc != 0:
            append_state(job, gpu, "fit", "FAILED", exit_code=fit_rc, log=str(fit_log))
            return
        checkpoint = select_checkpoint(output_root, fit_log, started_at)
        if checkpoint is None or not checkpoint.is_file():
            append_state(job, gpu, "checkpoint", "FAILED", output_root=str(output_root))
            return
    stamp = datetime.fromtimestamp(checkpoint.stat().st_mtime).isoformat(timespec="seconds")
    append_state(job, gpu, "checkpoint", "SELECTED", checkpoint=str(checkpoint.resolve()), mtime=stamp)

    test_log = log_path(job, gpu, "test", retry=log_failed(default_test))
    append_state(job, gpu, "test", "RUNNING", pid=os.getpid(), log=str(test_log), checkpoint=str(checkpoint.resolve()))
    test_command = [f"CUDA_VISIBLE_DEVICES={gpu}", sys.executable, "-m", "torchgeo", "test", "--config", str(job.test_config), f"--ckpt_path={checkpoint.resolve()}"]
    test_rc = run_with_tee(test_command, test_log)
    if test_rc != 0:
        append_state(job, gpu, "test", "FAILED", exit_code=test_rc, log=str(test_log), checkpoint=str(checkpoint.resolve()))
        return
    append_state(job, gpu, "pipeline", "SUCCESS", checkpoint=str(checkpoint.resolve()), fit_log=str(default_fit), test_log=str(test_log))


def worker(gpu: int, jobs: list[Job]) -> None:
    # 每项都捕获异常，保证该 GPU 可继续领取下一项；另一 GPU 完全独立。
    signal.signal(signal.SIGTERM, stop_active_child)
    signal.signal(signal.SIGINT, stop_active_child)
    for job in jobs:
        try:
            run_job(job, gpu)
        except Exception as exc:  # noqa: BLE001
            append_state(job, gpu, "pipeline", "FAILED", error=f"{type(exc).__name__}: {exc}")


def print_check() -> int:
    roots: dict[Path, Job] = {}
    print("序号\tDataset\tMethod\tExperiment\t训练配置\t测试配置\t训练\t测试\tdefault_root_dir\tcheckpoint\t日志\t计划GPU\t计划动作")
    blocked = 0
    for job in JOBS:
        problems, output_root = preflight(job)
        if output_root in roots:
            problems.append(f"输出目录与 {roots[output_root].stem} 冲突")
        roots[output_root] = job
        gpu = job.index % 2
        fit = log_path(job, gpu, "fit")
        test = log_path(job, gpu, "test")
        checkpoint = select_checkpoint(output_root, fit)
        if problems:
            action = "BLOCKED"
            blocked += 1
        elif checkpoint and contains_test_success(test):
            action = "跳过"
        elif checkpoint:
            action = "仅测试"
        else:
            action = "训练+测试"
        default_root = get_default_root(load_yaml(job.train_config)).relative_to(REPO)
        print("\t".join(map(str, [job.index + 1, job.dataset, job.method, job.experiment, job.train_config.relative_to(REPO), job.test_config.relative_to(REPO), job.train_config.is_file(), job.test_config.is_file(), default_root, "有效" if checkpoint else "无", "成功" if contains_test_success(test) else ("失败" if log_failed(test) else "无"), gpu, action])))
        for problem in problems:
            print(f"  BLOCKED: {problem}")
    return 1 if blocked else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="仅输出预检表，不启动任务")
    args = parser.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.check:
        return print_check()
    if print_check() != 0:
        return 1
    # 两个独立子进程各自串行执行；主进程 wait，正常退出时不会遗留 worker。
    queues = {0: [job for job in JOBS if job.index % 2 == 0], 1: [job for job in JOBS if job.index % 2 == 1]}
    workers = [Process(target=worker, args=(gpu, queue), daemon=False) for gpu, queue in queues.items()]
    for process in workers:
        process.start()
    try:
        for process in workers:
            process.join()
    except KeyboardInterrupt:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join()
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
