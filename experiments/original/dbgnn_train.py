from typing import Dict, Literal, cast

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

    result = {
        "best_epoch": float(best_epoch),
        "best_val_metric": float(best_val_metric),
        **{f"best_val_{k}": v for k, v in best_val_metrics.items()},
        **{f"best_test_{k}": v for k, v in best_test_metrics.items()},
    }
    return result


if __name__ == "__main__":
    # Configuration: Edit config dict or pass CLI args (CLI takes priority)
    config = {
        "dataset": "rel-f1",
        "task": "driver-position",
        "model": "sage_edge_attr",
        "tabular_model": "resnet",
        "seed": 42,
        "lr": 0.001,
        "min_epochs": 3,
        "batch_size": 32,
        "channels": 32,
        "num_layers": 2,
        "num_neighbors": 16,
        "max_steps_per_epoch": 10,
        "min_total_steps": 10,
        "aggr": "sum",
        "mlp_norm": "batch_norm",
        "cache_dir": ".cache",
        "process_bridge": True,
        "bridge_strategy": "keep_attributes",
        "process_hub": True,
        "hub_strategy": "keep_table",
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
            process_bridge=args.process_bridge,
            bridge_strategy=args.bridge_strategy,
            process_hub=args.process_hub,
            hub_strategy=args.hub_strategy,
        )
