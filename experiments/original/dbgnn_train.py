from typing import Dict, Literal, Optional, cast

import csv
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


def get_model(architecture: Literal["sage", "dbformer", "sage_edge_attr"], entity_table: str, **kwargs):
    if architecture == "sage":
        return SAGEModel(**kwargs)
    if architecture == "dbformer":
        return DBFormerModel(entity_table=entity_table, **kwargs)
    if architecture == "sage_edge_attr":
        return SAGEEdgeAttrModel(**kwargs)
    raise ValueError(f"Unknown architecture: {architecture}")


def print_heterodata_readable(data) -> None:
    print("\n=== HeteroData (readable) ===")
    print(data)

    # print("\nEdge stores:")
    # for edge_type in data.edge_types:
    #     store = data[edge_type]
    #     edge_index = getattr(store, "edge_index", None)
    #     num_edges = edge_index.size(1) if edge_index is not None else "N/A"
    #     print(f"  - {edge_type}: num_edges={num_edges}")

    #     edge_attr = getattr(store, "edge_attr", None)
    #     if torch.is_tensor(edge_attr):
    #         ea = cast(torch.Tensor, edge_attr)
    #         ea_f = ea.float()
    #         nan_count = int(torch.isnan(ea_f).sum())
    #         zero_rows = int((ea_f.abs().sum(dim=-1) == 0).sum())
    #         print(
    #             f"    edge_attr: shape={tuple(ea.shape)}, dtype={ea.dtype} | "
    #             f"min={ea_f.min():.4f}, max={ea_f.max():.4f}, mean={ea_f.mean():.4f} | "
    #             f"NaNs={nan_count}, all-zero rows={zero_rows}/{ea.size(0)}"
    #         )
    #     else:
    #         print("    edge_attr: none")

    print("=== End HeteroData ===\n")


def round_floats(value: object) -> object:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {k: round_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v) for v in value]
    return value


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
) -> None:
    with open(log_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    payload["epoch_history"].append(
        round_floats(
            {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "training_time_s": float(training_time_s),
            "val_metrics": {k: float(v) for k, v in val_metrics.items()},
            }
        )
    )

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

    payload["summary"] = round_floats({
        "best_epoch": int(best_epoch),
        "best_val_metric_name": best_val_metric_name,
        "best_val_metric_value": float(best_val_metric_value),
        "best_val_metrics": {k: float(v) for k, v in best_val_metrics.items()},
        "best_test_metrics": {k: float(v) for k, v in best_test_metrics.items()},
        "total_training_time_s": float(total_training_time_s),
    })

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


def run_training(
    dataset_name: str,
    task_name: str,
    model_architecture: Literal["sage", "dbformer", "sage_edge_attr"],
    tabular_model: str,
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
    log_dir: str = "logs/training_logs",
    log_filename: Optional[str] = None,
    toggle_logging: bool = False,
    summary_csv_path: str = "logs/training_logs/run_summary.csv",
    toggle_summary_csv: bool = True,
    process_bridge: bool = False,
    bridge_strategy: str = "default",
    process_hub: bool = False,
    hub_strategy: str = "default_combinations",
) -> Dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    print(f"Device: {device}")

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
    # TEST print - comment if not needed
    # print_heterodata_readable(data)
    
    entity_table = cast(str, getattr(task, "entity_table"))

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

    def train(split: str = "train") -> float:
        model.train()
        loader = loader_dict[split]

        loss_accum = 0.0
        count_accum = 0
        processed_steps = 0
        total_steps = max_steps_per_epoch

        for step, batch in enumerate(tqdm(loader, total=total_steps, desc=f"train/{split}"), start=1):
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

    run_params: Dict[str, object] = {
        "dataset": dataset_name,
        "task": task_name,
        "model_architecture": model_architecture,
        "tabular_model": tabular_model,
        "seed": seed,
        "lr": lr,
        "min_epochs": min_epochs,
        "batch_size": batch_size,
        "channels": channels,
        "num_layers": num_layers,
        "num_neighbors": num_neighbors,
        "max_steps_per_epoch": max_steps_per_epoch,
        "min_total_steps": min_total_steps,
        "aggr": aggr,
        "mlp_norm": mlp_norm,
        "log_dir": log_dir,
        "log_filename": log_filename,
        "process_bridge": process_bridge,
        "bridge_strategy": bridge_strategy,
        "process_hub": process_hub,
        "hub_strategy": hub_strategy,
    }

    filename = (log_filename or "").strip()
    if filename == "":
        filename = f"{dataset_name}_{task_name}_{model_architecture}_seed{seed}.json"
    elif not filename.endswith(".json"):
        filename = f"{filename}.json"

    log_path = os.path.join(log_dir, filename)
    if toggle_logging:
        init_training_log(log_path, run_params)

    val_table = task.get_table("val")
    training_time = 0.0

    best_val_metric = -math.inf if higher_is_better else math.inf
    best_epoch = -1
    best_val_metrics: Dict[str, float] = {}
    best_test_metrics: Dict[str, float] = {}

    for epoch in range(1, n_epochs + 1):
        start = timer()
        train_loss = train("train")
        training_time += timer() - start

        val_pred = evaluate("val")
        val_metrics = task.evaluate(val_pred, val_table, metrics=metrics)
        if toggle_logging:
            append_epoch_log(log_path, epoch, train_loss, training_time, val_metrics)

        current = val_metrics[tune_metric]
        improved = (higher_is_better and current >= best_val_metric) or (
            (not higher_is_better) and current <= best_val_metric
        )

        if improved:
            best_val_metric = current
            best_epoch = epoch
            best_val_metrics = {k: float(v) for k, v in val_metrics.items()}
            test_pred = evaluate("test")
            test_metrics = task.evaluate(test_pred, metrics=metrics)
            best_test_metrics = {k: float(v) for k, v in test_metrics.items()}

        val_str = " | ".join([f"val_{k}={v:.4f}" for k, v in val_metrics.items()])
        print(
            f"Epoch {epoch:03d}/{n_epochs} | loss={train_loss:.4f} | time={training_time:.1f}s | {val_str}"
        )

    print("\nBest validation epoch:", best_epoch)
    print("Best validation metrics:", best_val_metrics)
    print("Best corresponding test metrics:", best_test_metrics)

    if toggle_logging:
        finalize_training_log(
            log_path=log_path,
            best_epoch=best_epoch,
            best_val_metric_name=tune_metric,
            best_val_metric_value=best_val_metric,
            best_val_metrics=best_val_metrics,
            best_test_metrics=best_test_metrics,
            total_training_time_s=training_time,
        )
        print(f"Saved training log to: {log_path}")

    if toggle_summary_csv:
        summary_row: Dict[str, object] = {
            "dataset": dataset_name,
            "task": task_name,
            "model": model_architecture,
            "tabular_model": tabular_model,
            "seed": seed,
            "process_bridge": process_bridge,
            "bridge_strategy": bridge_strategy,
            "process_hub": process_hub,
            "hub_strategy": hub_strategy,
            "best_epoch": best_epoch,
            "training_time_s": float(training_time),
            **{f"best_val_{k}": float(v) for k, v in best_val_metrics.items()},
            **{f"best_test_{k}": float(v) for k, v in best_test_metrics.items()},
        }
        append_run_summary_csv(summary_csv_path, summary_row)
        print(f"Appended run summary to: {os.path.abspath(summary_csv_path)}")
    else:
        print("Summary CSV logging disabled (toggle_summary_csv=False).")

    result = {
        "best_epoch": float(best_epoch),
        "best_val_metric": float(best_val_metric),
        "training_time_s": float(training_time),
        **{f"best_val_{k}": v for k, v in best_val_metrics.items()},
        **{f"best_test_{k}": v for k, v in best_test_metrics.items()},
    }
    return result


if __name__ == "__main__":
    # Configuration: Edit config dict or pass CLI args (CLI takes priority)
    config = {
        "dataset": "rel-f1",
        "task": "driver-position",
        "model": "sage_edge_attr",          # "sage" | "dbformer" | "sage_edge_attr"
        "tabular_model": "resnet",          # "resnet" | "linear"
        "seed": 42,
        "lr": 0.001,
        "min_epochs": 10,
        "batch_size": 128,
        "channels": 64,
        "num_layers": 2,
        "num_neighbors": 32,
        "max_steps_per_epoch": 1000,
        "min_total_steps": 1000,
        "aggr": "sum",
        "mlp_norm": "batch_norm",

        "cache_dir": ".cache",
        "log_dir": "logs/training_logs",
        "log_filename": "benchmark_f1.json",
        "toggle_logging": False,
        "summary_csv_path": "logs/training_logs/run_summary_rel_f1.csv",
        "toggle_summary_csv": True,
        "process_bridge": True,
        "bridge_strategy": "keep_table",  # "default" | "keep_attributes" | "keep_table"
        "process_hub": True,
        "hub_strategy": "keep_attributes",       # "default_combinations" | "keep_attributes" | "keep_table"
    }
    
    # Parse CLI args (optional, defaults from config above)
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default=config["dataset"])
    parser.add_argument("--task", type=str, default=config["task"])
    parser.add_argument("--model", choices=["sage", "dbformer", "sage_edge_attr"], default=config["model"])
    parser.add_argument("--tabular_model", choices=["resnet", "linear"], default=config["tabular_model"])
    parser.add_argument("--seed", type=int, default=config["seed"])
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

    parser.add_argument("--cache_dir", type=str, default=config["cache_dir"])
    parser.add_argument("--log_dir", type=str, default=config["log_dir"])
    parser.add_argument("--log_filename", type=str, default=config["log_filename"])
    parser.add_argument("--toggle_logging", action="store_true", default=config["toggle_logging"])
    parser.add_argument("--no_toggle_logging", action="store_false", dest="toggle_logging")
    parser.add_argument("--summary_csv_path", type=str, default=config["summary_csv_path"])
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
        print(f"Skipping {args.dataset} - {args.task} (unsupported task type: {task.task_type})")
    else:
        run_training(
            dataset_name=args.dataset,
            task_name=args.task,
            model_architecture=cast(Literal["sage", "dbformer", "sage_edge_attr"], args.model),
            tabular_model=args.tabular_model,
            seed=args.seed,
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
            log_dir=args.log_dir,
            log_filename=args.log_filename,
            toggle_logging=args.toggle_logging,
            summary_csv_path=args.summary_csv_path,
            toggle_summary_csv=args.toggle_summary_csv,
            process_bridge=args.process_bridge,
            bridge_strategy=args.bridge_strategy,
            process_hub=args.process_hub,
            hub_strategy=args.hub_strategy,
        )
