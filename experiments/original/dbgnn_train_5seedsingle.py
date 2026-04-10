from typing import Any, Dict, Literal, Optional, Tuple, cast

import csv
import gc
import json
import math
import os
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from argparse import ArgumentParser
from timeit import default_timer as timer

import numpy as np
from tqdm import tqdm

import torch

from torch_geometric.loader import NeighborLoader

from relbench.base import TaskType
from relbench.tasks import get_task
from relbench.modeling.graph import get_node_train_table_input

from redelex.tasks.utils import is_temporal_task
from redelex.nn.models.sagegnn import SAGEModel
from redelex.nn.models.dbformer import DBFormerModel
from redelex.nn.models.sage_edge_attr import SAGEEdgeAttrModel

from experiments.utils import (
    get_cache_path,
    get_data_custom,
    get_loss,
    get_metrics,
    get_tune_metric,
)


def get_model(
    architecture: Literal["sage", "dbformer", "sage_edge_attr"], entity_table: str, **kwargs
):
    if architecture == "sage":
        return SAGEModel(**kwargs)
    if architecture == "dbformer":
        return DBFormerModel(entity_table=entity_table, **kwargs)
    if architecture == "sage_edge_attr":
        return SAGEEdgeAttrModel(**kwargs)
    raise ValueError(f"Unknown architecture: {architecture}")


def round_floats(value: object) -> object:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {k: round_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v) for v in value]
    return value


def cleanup_runtime_memory() -> None:
    """Free transient Python/Torch runtime memory without touching on-disk caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def init_training_log(log_path: str, run_params: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    payload = {
        "run_params": run_params,
        "epoch_history": [],
        "summary": {},
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_epoch_log(
    log_path: str,
    epoch: int,
    train_loss: float,
    training_time_s: float,
    val_metrics: Dict[str, float],
    test_metrics: Optional[Dict[str, float]] = None,
) -> None:
    with open(log_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    entry: Dict[str, object] = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "training_time_s": float(training_time_s),
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()}
        if test_metrics is not None
        else None,
    }
    payload["epoch_history"].append(round_floats(entry))

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def finalize_training_log(
    log_path: str,
    best_epoch: int,
    best_val_metric_name: str,
    best_val_metric_value: float,
    best_val_metrics: Dict[str, float],
    best_test_metrics: Dict[str, float],
    total_training_time_s: float,
) -> None:
    with open(log_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    payload["summary"] = round_floats(
        {
            "best_epoch": int(best_epoch),
            "best_val_metric_name": best_val_metric_name,
            "best_val_metric_value": float(best_val_metric_value),
            "best_val_metrics": {k: float(v) for k, v in best_val_metrics.items()},
            "best_test_metrics": {k: float(v) for k, v in best_test_metrics.items()},
            "total_training_time_s": float(total_training_time_s),
        }
    )

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_run_summary_csv(csv_path: str, row: Dict[str, object]) -> None:
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    normalized_row = cast(Dict[str, object], round_floats(row))

    existing_rows: list[Dict[str, str]] = []
    fieldnames: list[str] = list(normalized_row.keys())

    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            if reader.fieldnames is not None:
                fieldnames = list(reader.fieldnames)
                for key in normalized_row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)

    row_to_write = {key: "" for key in fieldnames}
    row_to_write.update({key: str(value) for key, value in normalized_row.items()})

    if existing_rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for existing in existing_rows:
                merged_existing = {key: existing.get(key, "") for key in fieldnames}
                writer.writerow(merged_existing)
            writer.writerow(row_to_write)
    else:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row_to_write)


def save_aggregated_summary_csv(
    csv_path: str,
    seeds: list,
    process_bridge: bool,
    bridge_strategy: str,
    process_hub: bool,
    hub_strategy: str,
    seed_results: list,
) -> None:
    """Save one aggregated row (mean/var per metric) across all seeds."""
    aggregate = build_aggregate_stats(seed_results)

    row: Dict[str, object] = {
        "seeds": str(seeds),
        "process_bridge": process_bridge,
        "bridge_strategy": bridge_strategy,
        "process_hub": process_hub,
        "hub_strategy": hub_strategy,
        **aggregate,
    }
    append_run_summary_csv(csv_path, row)
    print(f"Saved aggregated summary to: {os.path.abspath(csv_path)}")


def build_aggregate_stats(seed_results: list) -> Dict[str, object]:
    best_epochs = [int(r["best_epoch"]) for r in seed_results]
    train_times = np.array([r["training_time_s"] for r in seed_results], dtype=float)

    metric_keys = sorted(
        {
            k
            for r in seed_results
            for k in r.keys()
            if k.startswith("best_val_") or k.startswith("best_test_")
        }
    )

    stats: Dict[str, object] = {
        "best_epochs": str(best_epochs),
        "avg_training_time_s": round(float(np.mean(train_times)), 3),
    }

    for key in metric_keys:
        values = np.array([r[key] for r in seed_results if key in r], dtype=float)
        stats[f"{key}_mean"] = round(float(np.mean(values)), 4)
        stats[f"{key}_var"] = round(float(np.var(values)), 6)

    return stats


def save_multi_seed_log_json(
    json_path: str,
    run_params: Dict[str, object],
    seed_results: list,
) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    run_combo_params = {
        "process_bridge": run_params.get("process_bridge"),
        "bridge_strategy": run_params.get("bridge_strategy"),
        "process_hub": run_params.get("process_hub"),
        "hub_strategy": run_params.get("hub_strategy"),
    }
    payload: Dict[str, object] = {
        "run_params": run_combo_params,
        "seed_runs": [
            {
                "seed": int(r["seed"]),
                "epoch_history": r.get("epoch_history", []),
                "summary": r.get("summary", {}),
            }
            for r in seed_results
        ],
        "aggregate": build_aggregate_stats(seed_results),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(round_floats(payload), f, indent=2)
    print(f"Saved multi-seed log to: {os.path.abspath(json_path)}")


def save_sweep_manifest(
    sweep_dir: str,
    shared_params: Dict[str, object],
    run_configs: list[Dict[str, object]],
) -> list[str]:
    os.makedirs(sweep_dir, exist_ok=True)
    manifest_path = os.path.join(sweep_dir, "configs.json")
    config_ids: list[str] = [f"cfg_{idx}" for idx in range(1, len(run_configs) + 1)]

    manifest_configs: Dict[str, Dict[str, object]] = {}
    for cfg_id, cfg in zip(config_ids, run_configs):
        manifest_configs[cfg_id] = {
            "json_file": f"{cfg_id}.json",
            "process_bridge": bool(cfg["process_bridge"]),
            "bridge_strategy": str(cfg["bridge_strategy"]),
            "process_hub": bool(cfg["process_hub"]),
            "hub_strategy": str(cfg["hub_strategy"]),
        }

    manifest: Dict[str, object] = {
        **shared_params,
        "configs": manifest_configs,
    }

    try:
        with open(manifest_path, "x", encoding="utf-8") as f:
            json.dump(round_floats(manifest), f, indent=2)
        print(f"Saved sweep manifest: {os.path.abspath(manifest_path)}")
    except FileExistsError:
        print(f"Sweep manifest already exists (skipped): {os.path.abspath(manifest_path)}")
    return config_ids


def is_config_completed(json_path: str, expected_num_seeds: int) -> bool:
    """Return True when a config log has all expected seed runs with non-empty summaries."""
    if not os.path.exists(json_path):
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    seed_runs = payload.get("seed_runs", [])
    if not isinstance(seed_runs, list) or len(seed_runs) < expected_num_seeds:
        return False

    completed = 0
    for run in seed_runs:
        if not isinstance(run, dict):
            continue
        summary = run.get("summary")
        if isinstance(summary, dict) and len(summary) > 0:
            completed += 1

    return completed >= expected_num_seeds


def build_strategy_run_configs(
    run_all_configs: bool,
    process_bridge: bool,
    bridge_strategy: str,
    process_hub: bool,
    hub_strategy: str,
    bridge_strategy_options: list[str],
    hub_strategy_options: list[str],
) -> list[Dict[str, object]]:
    if not run_all_configs:
        return [
            {
                "process_bridge": process_bridge,
                "bridge_strategy": bridge_strategy,
                "process_hub": process_hub,
                "hub_strategy": hub_strategy,
            }
        ]

    # Keep option order stable while removing accidental duplicates.
    unique_bridge_strategies = list(dict.fromkeys(bridge_strategy_options))
    unique_hub_strategies = list(dict.fromkeys(hub_strategy_options))

    run_configs: list[Dict[str, object]] = []

    # 1) Default baseline: no bridge/hub processing.
    run_configs.append(
        {
            "process_bridge": False,
            "bridge_strategy": "default",
            "process_hub": False,
            "hub_strategy": "default_combinations",
        }
    )

    # 2) Hub-only runs: process_hub=True, process_bridge=False.
    for hub_opt in unique_hub_strategies:
        run_configs.append(
            {
                "process_bridge": False,
                "bridge_strategy": "default",
                "process_hub": True,
                "hub_strategy": hub_opt,
            }
        )

    # 3) Bridge-only runs: process_bridge=True, process_hub=False.
    for bridge_opt in unique_bridge_strategies:
        run_configs.append(
            {
                "process_bridge": True,
                "bridge_strategy": bridge_opt,
                "process_hub": False,
                "hub_strategy": "default_combinations",
            }
        )

    # 4) Full combinations: process_bridge=True and process_hub=True.
    for bridge_opt in unique_bridge_strategies:
        for hub_opt in unique_hub_strategies:
            run_configs.append(
                {
                    "process_bridge": True,
                    "bridge_strategy": bridge_opt,
                    "process_hub": True,
                    "hub_strategy": hub_opt,
                }
            )

    return run_configs


def get_default_16_combo_run_configs() -> list[Dict[str, object]]:
    """Canonical 16 combinations used by the sbatch array job order."""
    return build_strategy_run_configs(
        run_all_configs=True,
        process_bridge=False,
        bridge_strategy="default",
        process_hub=False,
        hub_strategy="default_combinations",
        bridge_strategy_options=["default", "keep_attributes", "keep_table"],
        hub_strategy_options=["default_combinations", "keep_attributes", "keep_table"],
    )


def resolve_config_id(
    run_configs: list[Dict[str, object]],
    process_bridge: bool,
    bridge_strategy: str,
    process_hub: bool,
    hub_strategy: str,
) -> str:
    for idx, cfg in enumerate(run_configs, start=1):
        if (
            bool(cfg["process_bridge"]) == process_bridge
            and str(cfg["bridge_strategy"]) == bridge_strategy
            and bool(cfg["process_hub"]) == process_hub
            and str(cfg["hub_strategy"]) == hub_strategy
        ):
            return f"cfg_{idx}"

    raise ValueError(
        "Unknown bridge/hub combination for cfg mapping: "
        f"process_bridge={process_bridge}, bridge_strategy={bridge_strategy}, "
        f"process_hub={process_hub}, hub_strategy={hub_strategy}"
    )


def build_training_data(
    dataset_name: str,
    task_name: str,
    cache_dir: str,
    process_bridge: bool,
    bridge_strategy: str,
    process_hub: bool,
    hub_strategy: str,
) -> Tuple[Any, Any, Dict[str, Any], str]:
    cache_path = get_cache_path(dataset_name, task_name, cache_dir)
    task, data, col_stats_dict = get_data_custom(
        dataset_name,
        task_name,
        str(cache_path),
        process_bridge=process_bridge,
        bridgeStrategy=bridge_strategy,
        process_hub=process_hub,
        hubStrategy=hub_strategy,
    )
    entity_table = cast(str, getattr(task, "entity_table"))
    return task, data, col_stats_dict, entity_table


def run_training(
    dataset_name: str,
    task_name: str,
    model_architecture: Literal["sage", "dbformer", "sage_edge_attr"],
    tabular_model: str,
    task: Any,
    data: Any,
    col_stats_dict: Dict[str, Any],
    entity_table: str,
    seed: int = 42,
    lr: float = 0.001,
    min_epochs: int = 10,
    batch_size: int = 512,
    channels: int = 64,
    num_layers: int = 2,
    num_neighbors: int = 32,
    max_steps_per_epoch: int = 2000,
    min_total_steps: int = 1000,
    aggr: str = "sum",
    mlp_norm: str = "batch_norm",
    cache_dir: str = ".cache",
    toggle_logging: bool = False,
    process_bridge: bool = False,
    bridge_strategy: str = "default",
    process_hub: bool = False,
    hub_strategy: str = "default_combinations",
    lr_decay_start_step: int = 0,
    lr_decay_steps: int = 0,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    print(f"Device: {device}")

    loss_fn, out_channels = get_loss(dataset_name, task_name)
    tune_metric, higher_is_better = get_tune_metric(dataset_name, task_name)
    metrics = get_metrics(dataset_name, task_name)

    is_temporal = is_temporal_task(task)

    loader_dict: Dict[str, NeighborLoader] = {}
    for split in ["train", "val", "test"]:
        table = task.get_table(split, mask_input_cols=False)
        table_input = get_node_train_table_input(table=table, task=task)
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[int(num_neighbors / 2**i) for i in range(num_layers)],
            time_attr="time" if is_temporal else None,
            input_nodes=table_input.nodes,
            input_time=table_input.time if is_temporal else None,
            transform=table_input.transform,
            batch_size=batch_size,
            temporal_strategy="uniform",
            shuffle=split == "train",
            num_workers=0,
            persistent_workers=False,
        )

    model = get_model(
        architecture=model_architecture,
        entity_table=entity_table,
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=num_layers,
        channels=channels,
        tabular_model=tabular_model,
        out_channels=out_channels,
        aggr=aggr,
        norm=mlp_norm,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if lr_decay_start_step < 0:
        raise ValueError("lr_decay_start_step must be >= 0")
    if lr_decay_steps < 0:
        raise ValueError("lr_decay_steps must be >= 0")

    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    if lr_decay_steps > 0:
        def lr_factor(step_idx: int) -> float:
            if step_idx < lr_decay_start_step:
                return 1.0

            decay_idx = step_idx - lr_decay_start_step
            if decay_idx >= lr_decay_steps:
                return 0.0

            return max(0.0, 1.0 - (decay_idx / float(lr_decay_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

    def train(split: str = "train") -> float:
        model.train()
        loader = loader_dict[split]

        loss_accum = 0.0
        count_accum = 0
        processed_steps = 0
        total_steps = max_steps_per_epoch

        for step, batch in enumerate(
            tqdm(loader, total=total_steps, desc=f"train/{split}"), start=1
        ):
            if step > max_steps_per_epoch:
                break

            batch = batch.to(device)

            optimizer.zero_grad()
            pred = model(batch, entity_table)
            pred = pred.view(-1) if pred.size(1) == 1 else pred

            if pred.size(0) != batch[entity_table].batch_size:
                pred = pred[: batch[entity_table].batch_size]

            if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                target = batch[entity_table].y.long()
            else:
                target = batch[entity_table].y.float()

            loss = loss_fn(pred.float(), target)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            loss_accum += loss.detach().item() * pred.size(0)
            count_accum += pred.size(0)
            processed_steps = step

        print(f"train/{split} steps this epoch: {processed_steps}")
        return loss_accum / count_accum if count_accum > 0 else 0.0

    @torch.no_grad()
    def evaluate(split: str) -> np.ndarray:
        model.eval()
        loader = loader_dict[split]

        pred_list = []
        for batch in tqdm(loader, desc=f"eval/{split}"):
            batch = batch.to(device)
            pred = model(batch, entity_table)

            if task.task_type in [
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
            ]:
                pred = torch.sigmoid(pred)
            elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                pred = torch.softmax(pred, dim=1)

            pred = pred.view(-1) if pred.size(1) == 1 else pred

            if pred.size(0) != batch[entity_table].batch_size:
                pred = pred[: batch[entity_table].batch_size]

            pred_list.append(pred.detach().cpu())

        return torch.cat(pred_list, dim=0).numpy()

    epoch_steps = min(len(loader_dict["train"]), max_steps_per_epoch)
    n_epochs = max(math.ceil(min_total_steps / max(epoch_steps, 1)), min_epochs)

    print(
        f"Start training | dataset={dataset_name} task={task_name} model={model_architecture} "
        f"epochs={n_epochs} batch={batch_size} neighbors={num_neighbors}"
    )
    if lr_decay_steps > 0:
        print(
            f"LR scheduler: linear decay starts at step {lr_decay_start_step} "
            f"over {lr_decay_steps} steps"
        )

    epoch_history: list[Dict[str, object]] = []

    val_table = task.get_table("val")
    training_time = 0.0

    best_val_metric = -math.inf if higher_is_better else math.inf
    best_epoch = -1
    best_val_metrics: Dict[str, float] = {}
    best_test_metrics: Dict[str, float] = {}

    for epoch in range(1, n_epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None and current_lr == 0.0:
            print(f"Early stop at epoch {epoch}: learning rate reached 0.")
            break
        start = timer()
        train_loss = train("train")
        training_time += timer() - start

        val_pred = evaluate("val")
        val_metrics = task.evaluate(val_pred, val_table, metrics=metrics)

        current = val_metrics[tune_metric]
        improved = (higher_is_better and current >= best_val_metric) or (
            (not higher_is_better) and current <= best_val_metric
        )

        epoch_test_metrics: Optional[Dict[str, float]] = None
        if toggle_logging:
            test_pred = evaluate("test")
            test_metrics = task.evaluate(test_pred, metrics=metrics)
            epoch_test_metrics = {k: float(v) for k, v in test_metrics.items()}
            epoch_entry = cast(
                Dict[str, object],
                round_floats(
                    {
                        "epoch": epoch,
                        "lr": current_lr,
                        "train_loss": float(train_loss),
                        "training_time_s": float(training_time),
                        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
                        "test_metrics": epoch_test_metrics,
                    }
                ),
            )
            epoch_entry["lr"] = round(current_lr, 6)
            epoch_history.append(epoch_entry)

        if improved:
            best_val_metric = current
            best_epoch = epoch
            best_val_metrics = {k: float(v) for k, v in val_metrics.items()}
            if epoch_test_metrics is None:
                test_pred = evaluate("test")
                test_metrics = task.evaluate(test_pred, metrics=metrics)
                epoch_test_metrics = {k: float(v) for k, v in test_metrics.items()}
            best_test_metrics = dict(epoch_test_metrics)

        val_str = " | ".join([f"val_{k}={v:.4f}" for k, v in val_metrics.items()])
        print(
            f"Epoch {epoch:03d}/{n_epochs} | lr={current_lr:.8f} ({current_lr:.3e}) | "
            f"loss={train_loss:.4f} | time={training_time:.1f}s | {val_str}"
        )

    print("\nBest validation epoch:", best_epoch)
    print("Best validation metrics:", best_val_metrics)
    print("Best corresponding test metrics:", best_test_metrics)

    summary_payload: Dict[str, object] = cast(
        Dict[str, object],
        round_floats(
            {
                "best_epoch": int(best_epoch),
                "best_val_metric_name": tune_metric,
                "best_val_metric_value": float(best_val_metric),
                "best_val_metrics": {k: float(v) for k, v in best_val_metrics.items()},
                "best_test_metrics": {k: float(v) for k, v in best_test_metrics.items()},
                "total_training_time_s": float(training_time),
            }
        ),
    )

    result = {
        "seed": int(seed),
        "best_epoch": float(best_epoch),
        "training_time_s": float(training_time),
        "epoch_history": epoch_history,
        "summary": summary_payload,
        **{f"best_val_{k}": v for k, v in best_val_metrics.items()},
        **{f"best_test_{k}": v for k, v in best_test_metrics.items()},
    }

    del model
    del optimizer
    if scheduler is not None:
        del scheduler
    del loader_dict
    cleanup_runtime_memory()

    return result


if __name__ == "__main__":
    # Configuration: Edit config dict or pass CLI args (CLI takes priority)
    config = {
        "dataset": "rel-f1",
        "task": "driver-position",
        "model": "sage_edge_attr",  # "sage" | "dbformer" | "sage_edge_attr"
        "tabular_model": "resnet",  # "resnet" | "linear"
        "seed": 42,
        "seeds": [],#[42, 43, 44, 45, 46],
        "lr": 0.1,                      # 0.001
        "min_epochs": 2,               # 10
        "batch_size": 64,              # 128
        "channels": 16,                 # 64
        "num_layers": 2,                # 2
        "num_neighbors": 16,            # 32
        "max_steps_per_epoch": 1000,    # 1000
        "min_total_steps": 100,        # 1000 + lr_decay_steps
        "aggr": "sum",
        "mlp_norm": "batch_norm",
        "lr_decay_start_step": 50,    # 1000
        "lr_decay_steps": 50,         # 3000
        "cache_dir": ".cache",
        "log_dir": "logs/training_logs",
        "toggle_logging": True,
        "toggle_summary_csv": True,
        "process_bridge": True,
        "bridge_strategy": "keep_attributes",  # "default" | "keep_attributes" | "keep_table"
        "process_hub": False,
        "hub_strategy": "default_combinations",  # "default_combinations" | "keep_attributes" | "keep_table"
    }

    # Parse CLI args (optional, defaults from config above)
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config["dataset"])
    parser.add_argument("--task", type=str, default=config["task"])
    parser.add_argument("--model", choices=["sage", "dbformer", "sage_edge_attr"], default=config["model"])
    parser.add_argument("--tabular_model", choices=["resnet", "linear"], default=config["tabular_model"])
    parser.add_argument("--seed", type=int, default=config["seed"])
    parser.add_argument("--seeds", type=int, nargs="+", default=config["seeds"])
    parser.add_argument("--lr", type=float, default=config["lr"])
    parser.add_argument("--min_epochs", type=int, default=config["min_epochs"])
    parser.add_argument("--batch_size", type=int, default=config["batch_size"])
    parser.add_argument("--channels", type=int, default=config["channels"])
    parser.add_argument("--num_layers", type=int, default=config["num_layers"])
    parser.add_argument("--num_neighbors", type=int, default=config["num_neighbors"])
    parser.add_argument("--max_steps_per_epoch", type=int, default=config["max_steps_per_epoch"])
    parser.add_argument("--min_total_steps", type=int, default=config["min_total_steps"])
    parser.add_argument("--aggr", choices=["sum", "mean", "max"], default=config["aggr"])
    parser.add_argument("--mlp_norm", choices=["batch_norm", "layer_norm"], default=config["mlp_norm"])
    parser.add_argument("--lr_decay_start_step", type=int, default=config["lr_decay_start_step"])
    parser.add_argument("--lr_decay_steps", type=int, default=config["lr_decay_steps"])

    parser.add_argument("--cache_dir", type=str, default=config["cache_dir"])
    parser.add_argument("--log_dir", type=str, default=config["log_dir"])
    parser.add_argument("--toggle_logging", action="store_true", default=config["toggle_logging"])
    parser.add_argument("--no_toggle_logging", action="store_false", dest="toggle_logging")
    parser.add_argument("--toggle_summary_csv", action="store_true", default=config["toggle_summary_csv"])
    parser.add_argument("--no_toggle_summary_csv", action="store_false", dest="toggle_summary_csv")

    parser.add_argument("--process_bridge", action="store_true", default=config["process_bridge"])
    parser.add_argument("--bridge_strategy", type=str, default=config["bridge_strategy"])
    parser.add_argument("--process_hub", action="store_true", default=config["process_hub"])
    parser.add_argument("--hub_strategy", type=str, default=config["hub_strategy"])

    args = parser.parse_args()
    print(f"Using config: {args}")

    task = get_task(args.dataset, args.task)
    if task.task_type in [TaskType.LINK_PREDICTION, TaskType.MULTILABEL_CLASSIFICATION]:
        print(
            f"Skipping {args.dataset} - {args.task} (unsupported task type: {task.task_type})"
        )
    else:
        seeds = args.seeds if len(args.seeds) > 0 else [args.seed]
        print(f"Running seeds: {seeds}")

        process_bridge = args.process_bridge
        bridge_strategy = args.bridge_strategy
        process_hub = args.process_hub
        hub_strategy = args.hub_strategy

        base_filename = f"{args.dataset}_{args.task}_{args.model}"
        sweep_dir = os.path.join(args.log_dir, base_filename)
        run_configs = get_default_16_combo_run_configs()
        cfg_id = resolve_config_id(
            run_configs=run_configs,
            process_bridge=process_bridge,
            bridge_strategy=bridge_strategy,
            process_hub=process_hub,
            hub_strategy=hub_strategy,
        )
        csv_path = os.path.join(args.log_dir, base_filename, f"{cfg_id}.csv")

        shared_manifest_params: Dict[str, object] = {
            "dataset": args.dataset,
            "task": args.task,
            "model_architecture": args.model,
            "tabular_model": args.tabular_model,
            "seeds": seeds,
            "lr": args.lr,
            "min_epochs": args.min_epochs,
            "batch_size": args.batch_size,
            "channels": args.channels,
            "num_layers": args.num_layers,
            "num_neighbors": args.num_neighbors,
            "max_steps_per_epoch": args.max_steps_per_epoch,
            "min_total_steps": args.min_total_steps,
            "aggr": args.aggr,
            "mlp_norm": args.mlp_norm,
            "lr_decay_start_step": args.lr_decay_start_step,
            "lr_decay_steps": args.lr_decay_steps,
        }

        combined_run_params: Dict[str, object] = {
            "process_bridge": process_bridge,
            "bridge_strategy": bridge_strategy,
            "process_hub": process_hub,
            "hub_strategy": hub_strategy,
        }

        json_path: Optional[str] = None
        if args.toggle_logging:
            save_sweep_manifest(
                sweep_dir=sweep_dir,
                shared_params=shared_manifest_params,
                run_configs=run_configs,
            )
            json_path = os.path.join(sweep_dir, f"{cfg_id}.json")
            print(
                f"Config mapping: process_bridge={process_bridge}, "
                f"bridge_strategy={bridge_strategy}, process_hub={process_hub}, "
                f"hub_strategy={hub_strategy} -> {cfg_id}.json"
            )
            save_multi_seed_log_json(
                json_path=json_path,
                run_params=combined_run_params,
                seed_results=[],
            )

        print(
            f"\n=== Configuration | "
            f"process_bridge={process_bridge} bridge_strategy={bridge_strategy} "
            f"process_hub={process_hub} hub_strategy={hub_strategy} ==="
        )

        print("\n=== Building training data ===")
        task, data, col_stats_dict, entity_table = build_training_data(
            dataset_name=args.dataset,
            task_name=args.task,
            cache_dir=args.cache_dir,
            process_bridge=process_bridge,
            bridge_strategy=bridge_strategy,
            process_hub=process_hub,
            hub_strategy=hub_strategy,
        )

        seed_results: list[Dict[str, Any]] = []
        for seed in seeds:
            print(f"\n=== Running seed {seed} ===")
            result = run_training(
                dataset_name=args.dataset,
                task_name=args.task,
                model_architecture=cast(
                    Literal["sage", "dbformer", "sage_edge_attr"], args.model
                ),
                tabular_model=args.tabular_model,
                task=task,
                data=data,
                col_stats_dict=col_stats_dict,
                entity_table=entity_table,
                seed=seed,
                lr=args.lr,
                min_epochs=args.min_epochs,
                batch_size=args.batch_size,
                channels=args.channels,
                num_layers=args.num_layers,
                num_neighbors=args.num_neighbors,
                max_steps_per_epoch=args.max_steps_per_epoch,
                min_total_steps=args.min_total_steps,
                aggr=args.aggr,
                mlp_norm=args.mlp_norm,
                cache_dir=args.cache_dir,
                toggle_logging=args.toggle_logging,
                process_bridge=process_bridge,
                bridge_strategy=bridge_strategy,
                process_hub=process_hub,
                hub_strategy=hub_strategy,
                lr_decay_start_step=args.lr_decay_start_step,
                lr_decay_steps=args.lr_decay_steps,
            )
            seed_results.append(result)

            if args.toggle_logging and json_path is not None:
                save_multi_seed_log_json(
                    json_path=json_path,
                    run_params=combined_run_params,
                    seed_results=seed_results,
                )

            cleanup_runtime_memory()

        if len(seed_results) > 0:
            print("\n=== Multi-seed summary ===")
            best_epochs = [int(r["best_epoch"]) for r in seed_results]
            train_times = np.array(
                [r["training_time_s"] for r in seed_results], dtype=float
            )

            print(
                " | ".join(
                    [
                        f"runs={len(seed_results)}",
                        f"best_epochs={best_epochs}",
                        f"avg_training_time_s={np.mean(train_times):.2f}",
                    ]
                )
            )

            val_metric_keys = sorted(
                {k for r in seed_results for k in r.keys() if k.startswith("best_val_")}
            )
            test_metric_keys = sorted(
                {k for r in seed_results for k in r.keys() if k.startswith("best_test_")}
            )

            if len(val_metric_keys) > 0:
                print("Validation metrics (mean/var):")
                for key in val_metric_keys:
                    values = np.array(
                        [r[key] for r in seed_results if key in r], dtype=float
                    )
                    if values.size > 0:
                        print(
                            f"  {key}: mean={np.mean(values):.3f}, var={np.var(values):.3f}"
                        )

            if len(test_metric_keys) > 0:
                print("Test metrics (mean/var):")
                for key in test_metric_keys:
                    values = np.array(
                        [r[key] for r in seed_results if key in r], dtype=float
                    )
                    if values.size > 0:
                        print(
                            f"  {key}: mean={np.mean(values):.3f}, var={np.var(values):.3f}"
                        )

            if args.toggle_summary_csv:
                save_aggregated_summary_csv(
                    csv_path=csv_path,
                    seeds=seeds,
                    process_bridge=process_bridge,
                    bridge_strategy=bridge_strategy,
                    process_hub=process_hub,
                    hub_strategy=hub_strategy,
                    seed_results=seed_results,
                )

        del seed_results
        del task
        del data
        del col_stats_dict
        del entity_table
        cleanup_runtime_memory()
