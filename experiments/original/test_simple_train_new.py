from typing import Any, Dict, Literal

import os
import random
from timeit import default_timer as timer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
from tqdm import tqdm
import torch

from torch_frame.data import StatType
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from relbench.base import BaseTask, EntityTask, TaskType
from relbench.tasks import get_task
from relbench.modeling.graph import get_node_train_table_input

from redelex.tasks import CTUEntityTaskTemporal
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


def simple_train_test(
    dataset_name: str,
    task_name: str,
    model_architecture: str = "sage",
    tabular_model: str = "resnet",
    random_seed: int = 42,
    lr: float = 0.001,
    num_epochs: int = 5,
    batch_size: int = 64,
    channels: int = 64,
    num_layers: int = 2,
    num_neighbors: int = 16,
    max_steps_per_epoch: int = 10,
    aggr: str = "sum",
    norm: str = "batch_norm",
    process_bridge: bool = False,
    bridge_strategy: str = "default",
    process_hub: bool = False,
    hub_strategy: str = "default_combinations",
    cache_dir: str = ".cache",
):
    """
    Simple training test function.
    
    Args:
        dataset_name: Name of the dataset
        task_name: Name of the task
        model_architecture: Model architecture ('sage', 'dbformer', 'sage_edge_attr')
        tabular_model: Tabular model type ('resnet', 'linear')
        random_seed: Random seed for reproducibility
        lr: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        channels: Number of hidden channels
        num_layers: Number of GNN layers
        num_neighbors: Number of neighbors to sample
        max_steps_per_epoch: Maximum training steps per epoch
        aggr: Aggregation function ('sum', 'mean', 'max')
        norm: Normalization type ('batch_norm', 'layer_norm')
        process_bridge: Whether to process bridge tables
        bridge_strategy: Strategy for bridge table processing
        process_hub: Whether to process hub tables
        hub_strategy: Strategy for hub table processing
        cache_dir: Directory for caching data
    """
    
    # Set seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    cache_path = get_cache_path(dataset_name, task_name, cache_dir)
    task, data, col_stats_dict = get_data_custom(
        dataset_name,
        task_name,
        cache_path,
        process_bridge=process_bridge,
        bridgeStrategy=bridge_strategy,
        process_hub=process_hub,
        hubStrategy=hub_strategy
    )

    print(f"Dataset: {dataset_name}, Task: {task_name}")
    print(f"Entity table: {task.entity_table}")
    print(f"Task type: {task.task_type}")
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")

    # Check for edge attributes
    edge_types_with_attr = []
    for edge_type in data.edge_types:
        if hasattr(data[edge_type], 'edge_attr'):
            edge_attr_shape = data[edge_type].edge_attr.shape if hasattr(data[edge_type].edge_attr, 'shape') else 'unknown'
            edge_types_with_attr.append((edge_type, edge_attr_shape))
    print(f"Edge types with attributes: {edge_types_with_attr}")

    loss_fn, out_channels = get_loss(dataset_name, task_name)
    tune_metric, higher_is_better = get_tune_metric(dataset_name, task_name)
    metrics = get_metrics(dataset_name, task_name)

    is_temporal = isinstance(task, CTUEntityTaskTemporal) or isinstance(task, EntityTask)

    # Create data loaders
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

    # Create model
    if model_architecture == "sage":
        model = SAGEModel(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=num_layers,
            channels=channels,
            tabular_model=tabular_model,
            out_channels=out_channels,
            aggr=aggr,
            norm=norm,
        )
    elif model_architecture == "dbformer":
        model = DBFormerModel(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=num_layers,
            channels=channels,
            tabular_model=tabular_model,
            out_channels=out_channels,
            aggr=aggr,
            norm=norm,
            entity_table=task.entity_table,
        )
    elif model_architecture == "sage_edge_attr":
        model = SAGEEdgeAttrModel(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=num_layers,
            channels=channels,
            tabular_model=tabular_model,
            out_channels=out_channels,
            aggr=aggr,
            norm=norm,
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_architecture}")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(split: str = "train") -> float:
        model.train()
        loader = loader_dict[split]
        loss_accum = count_accum = 0
        steps = 0

        for batch in tqdm(loader, desc=f"Training ({split})", total=min(len(loader), max_steps_per_epoch)):
            batch = batch.to(device)

            optimizer.zero_grad()
            pred = model(batch, task.entity_table)
            pred = pred.view(-1) if pred.size(1) == 1 else pred

            if pred.size(0) != batch[task.entity_table].batch_size:
                pred = pred[: batch[task.entity_table].batch_size]

            if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                target = batch[task.entity_table].y.long()
            else:
                target = batch[task.entity_table].y.float()

            loss = loss_fn(pred.float(), target)
            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item() * pred.size(0)
            count_accum += pred.size(0)

            steps += 1
            if steps > max_steps_per_epoch:
                break

        return loss_accum / count_accum if count_accum > 0 else 0.0

    @torch.no_grad()
    def test(split: str) -> np.ndarray:
        loader = loader_dict[split]
        model.eval()

        pred_list = []
        for batch in tqdm(loader, desc=f"Testing ({split})"):
            batch = batch.to(device)
            pred = model(batch, task.entity_table)

            if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
                pred = torch.sigmoid(pred)
            elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                pred = torch.softmax(pred, dim=1)

            pred = pred.view(-1) if pred.size(1) == 1 else pred

            if pred.size(0) != batch[task.entity_table].batch_size:
                pred = pred[: batch[task.entity_table].batch_size]

            pred_list.append(pred.detach().cpu())
        return torch.cat(pred_list, dim=0).numpy()

    print(f"\nStarting training for {num_epochs} epochs...")

    val_table = task.get_table("val")

    for epoch in range(1, num_epochs + 1):
        start_time = timer()
        train_loss = train("train")
        end_time = timer()
        train_time = end_time - start_time

        print(f"Epoch {epoch}/{num_epochs} - Train loss: {train_loss:.4f} - Time: {train_time:.2f}s")

        # Evaluate on validation set
        val_pred = test("val")
        val_results = task.evaluate(val_pred, val_table, metrics=metrics)

        print("Validation results:")
        for metric_name, value in val_results.items():
            print(f"  {metric_name}: {value:.4f}")
        print()

    # Final evaluation on test set
    print("Final evaluation on test set:")
    test_pred = test("test")
    test_results = task.evaluate(test_pred, metrics=metrics)

    for metric_name, value in test_results.items():
        print(f"  {metric_name}: {value:.4f}")

    print("\nTraining completed!")


if __name__ == "__main__":
    simple_train_test(
        dataset_name="rel-f1",
        task_name="driver-position",
        model_architecture="sage_edge_attr",
        tabular_model="resnet",
        num_epochs=20,
        batch_size=32,
        channels=128,
        num_layers=2,
        process_bridge=True,
        bridge_strategy="keep_attributes",
        process_hub=True,
        hub_strategy="keep_table",
    )
