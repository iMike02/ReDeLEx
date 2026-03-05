from typing import Any, Dict, Literal, Optional

import math
import os
import random
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from argparse import ArgumentParser
from datetime import datetime
from timeit import default_timer as timer

import numpy as np

from tqdm import tqdm

import torch

from torch_frame.data import StatType

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader


from relbench.base import BaseTask, EntityTask, TaskType
from relbench.tasks import get_task

from redelex.data import get_node_train_table_input
from redelex.tasks import CTUBaseEntityTask, CTUEntityTaskTemporal
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

# Ray imports (optional, only needed for tuning)
try:
    import ray
    from ray import tune
    from ray import train as ray_train
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.logger.mlflow import MLflowLoggerCallback
    from ray.tune.logger.aim import AimLoggerCallback
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


def get_model(architecture: Literal["sage", "dbformer", "sage_edge_attr"], entity_table: str, **kwargs):
    if architecture == "sage":
        return SAGEModel(**kwargs)
    elif architecture == "dbformer":
        return DBFormerModel(entity_table=entity_table, **kwargs)
    elif architecture == "sage_edge_attr":
        return SAGEEdgeAttrModel(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def run_training(
    config: Dict[str, Any],
    data: Optional[HeteroData] = None,
    task: Optional[BaseTask] = None,
    col_stats_dict: Optional[Dict] = None,
    use_ray: bool = False,
):
    """
    Core training function that works with both standalone and Ray Tune modes.
    
    Args:
        config: Configuration dictionary
        data: Pre-loaded HeteroData (for Ray Tune to avoid reloading)
        task: Pre-loaded task (for Ray Tune)
        col_stats_dict: Pre-loaded column statistics (for Ray Tune)
        use_ray: Whether to report to Ray Tune
    """
    # Extract config values
    dataset_name = config["dataset"]
    task_name = config["task"]
    model_architecture = config["model"]
    tabular_model = config["tabular_model"]
    random_seed = config["seed"]
    lr = config["lr"]
    num_epochs = config["epochs"]
    batch_size = config["batch_size"]
    channels = config["channels"]
    num_layers = config["num_layers"]
    num_neighbors = config["num_neighbors"]
    max_steps_per_epoch = config["max_steps_per_epoch"]
    aggr = config["aggr"]
    norm = config["norm"]
    process_bridge = config.get("process_bridge", False)
    bridge_strategy = config.get("bridge_strategy", "default")
    process_hub = config.get("process_hub", False)
    hub_strategy = config.get("hub_strategy", "default_combinations")
    cache_dir = config.get("cache_dir", ".cache")

    # Set seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not use_ray:
        print(f"Using device: {device}")

    # Load data if not provided (standalone mode)
    if data is None or task is None or col_stats_dict is None:
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

        if not use_ray:
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
    model = get_model(
        architecture=model_architecture,
        entity_table=task.entity_table,
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=num_layers,
        channels=channels,
        tabular_model=tabular_model,
        out_channels=out_channels,
        aggr=aggr,
        norm=norm,
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(split: str = "train") -> float:
        model.train()
        loader = loader_dict[split]
        loss_accum = count_accum = 0
        steps = 0

        desc = f"Training ({split})" if not use_ray else None
        for batch in tqdm(loader, desc=desc, total=min(len(loader), max_steps_per_epoch), disable=use_ray):
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
        desc = f"Testing ({split})" if not use_ray else None
        for batch in tqdm(loader, desc=desc, disable=use_ray):
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

    # Training loop
    if not use_ray:
        print(f"\nStarting training for {num_epochs} epochs...")

    val_table = task.get_table("val")
    best_val_metric = -math.inf if higher_is_better else math.inf
    training_time = 0

    # Initial report for Ray
    if use_ray:
        ray_train.report({
            f"val_{tune_metric}": best_val_metric,
            f"test_{tune_metric}": best_val_metric
        })

    for epoch in range(1, num_epochs + 1):
        start_time = timer()
        train_loss = train("train")
        end_time = timer()
        train_time = end_time - start_time
        training_time += train_time

        if not use_ray:
            print(f"Epoch {epoch}/{num_epochs} - Train loss: {train_loss:.4f} - Time: {train_time:.2f}s")

        # Evaluate on validation set
        val_pred = test("val")
        val_results = task.evaluate(val_pred, val_table, metrics=metrics)
        
        if not use_ray:
            print("Validation results:")
            for metric_name, value in val_results.items():
                print(f"  {metric_name}: {value:.4f}")

        # Prepare metrics for reporting
        metrics_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_time": training_time,
            **{f"val_{k}": v for k, v in val_results.items()},
        }

        # Update best metric
        if (higher_is_better and val_results[tune_metric] >= best_val_metric) or \
           (not higher_is_better and val_results[tune_metric] <= best_val_metric):
            best_val_metric = val_results[tune_metric]

        # Report to Ray Tune or print
        if use_ray:
            ray_train.report(metrics_dict)
        else:
            print()

    # Final evaluation on test set
    if not use_ray:
        print("\nFinal evaluation on test set:")
    
    test_pred = test("test")
    test_results = task.evaluate(test_pred, metrics=metrics)
    
    if not use_ray:
        for metric_name, value in test_results.items():
            print(f"  {metric_name}: {value:.4f}")
        print("\nTraining completed!")
    
    return {
        "best_val_metric": best_val_metric,
        "test_results": test_results,
    }


def simple_train_test(**kwargs):
    """Wrapper for standalone training (backward compatibility)"""
    # Convert old parameter names to new config format
    config = {
        "dataset": kwargs.get("dataset_name"),
        "task": kwargs.get("task_name"),
        "model": kwargs.get("model_architecture", "sage"),
        "tabular_model": kwargs.get("tabular_model", "resnet"),
        "seed": kwargs.get("random_seed", 42),
        "lr": kwargs.get("lr", 0.001),
        "epochs": kwargs.get("num_epochs", 5),
        "batch_size": kwargs.get("batch_size", 64),
        "channels": kwargs.get("channels", 64),
        "num_layers": kwargs.get("num_layers", 2),
        "num_neighbors": kwargs.get("num_neighbors", 16),
        "max_steps_per_epoch": kwargs.get("max_steps_per_epoch", 10),
        "aggr": kwargs.get("aggr", "sum"),
        "norm": kwargs.get("norm", "batch_norm"),
        "process_bridge": kwargs.get("process_bridge", False),
        "bridge_strategy": kwargs.get("bridge_strategy", "default"),
        "process_hub": kwargs.get("process_hub", False),
        "hub_strategy": kwargs.get("hub_strategy", "default_combinations"),
        "cache_dir": kwargs.get("cache_dir", ".cache"),
    }
    
    return run_training(config, use_ray=False)


def run_ray_tuner(
    dataset_name: str,
    task_name: str,
    model_architecture: str = "sage",
    tabular_model: str = "resnet",
    random_seed: int = 42,
    ray_address: Optional[str] = "local",
    ray_storage_path: Optional[str] = None,
    ray_experiment_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    mlflow_experiment: str = "gnn_experiment",
    aim_repo: Optional[str] = None,
    num_samples: int = 1,
    num_gpus: float = 0,
    num_cpus: int = 1,
    cache_dir: str = ".cache",
    process_bridge: bool = False,
    bridge_strategy: str = "default",
    process_hub: bool = False,
    hub_strategy: str = "default_combinations",
    tune_config: Optional[Dict] = None,
):
    """Run hyperparameter tuning with Ray Tune"""
    
    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune is not installed. Install with: pip install ray[tune]")

    # Set seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Initialize Ray
    if num_gpus > 0 and ray_address == "local":
        try:
            from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
            nvmlInit()
            free_memory = [
                int(nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free)
                for i in range(torch.cuda.device_count())
            ]
            device_idx = np.argsort(free_memory)[::-1]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_idx[:int(num_gpus)].astype(str))
            print(f"Free memory: {free_memory}, Using devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
        except ImportError:
            print("pynvml not available, using default GPU selection")

    ray.init(
        address=ray_address,
        ignore_reinit_error=True,
        log_to_driver=False,
        include_dashboard=False,
        num_cpus=num_cpus if ray_address == "local" else None,
        num_gpus=int(num_gpus) if ray_address == "local" else None,
    )

    # Load data once (shared across trials)
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

    # Define search space (can be overridden with tune_config)
    default_search_space = {
        "dataset": dataset_name,
        "task": task_name,
        "model": model_architecture,
        "tabular_model": tabular_model,
        "seed": tune.randint(0, 1000),
        # Training config
        "epochs": 10,
        "max_steps_per_epoch": 2000,
        "lr": tune.choice([0.001, 0.005, 0.01]),
        "batch_size": tune.choice([128, 256, 512]),
        # Model config
        "channels": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3, 4]),
        "num_neighbors": tune.choice([16, 32, 64]),
        "aggr": tune.choice(["sum", "mean", "max"]),
        "norm": tune.choice(["batch_norm", "layer_norm"]),
        # Processing config
        "process_bridge": process_bridge,
        "bridge_strategy": bridge_strategy,
        "process_hub": process_hub,
        "hub_strategy": hub_strategy,
        "cache_dir": cache_dir,
    }

    # Merge with custom tune config if provided
    if tune_config:
        default_search_space.update(tune_config)

    search_space = default_search_space

    # Get tune metric
    metric, higher_is_better = get_tune_metric(dataset_name, task_name)
    tune_metric_name = f"val_{metric}"
    metric_mode = "max" if higher_is_better else "min"

    # Setup scheduler
    scheduler = ASHAScheduler(
        max_t=search_space["epochs"],
        grace_period=3,
        reduction_factor=2
    )

    # Setup callbacks
    ray_callbacks = []
    if mlflow_uri:
        ray_callbacks.append(
            MLflowLoggerCallback(
                tracking_uri=mlflow_uri,
                experiment_name=mlflow_experiment,
            )
        )
    if aim_repo:
        ray_callbacks.append(
            AimLoggerCallback(
                repo=aim_repo,
                experiment_name=ray_experiment_name or f"tune_{dataset_name}_{task_name}"
            )
        )

    # Experiment name
    if ray_experiment_name is None:
        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        ray_experiment_name = f"tune_{time}_{dataset_name}_{task_name}"

    # Calculate GPU resources
    gpus_per_trial = 0
    if num_gpus > 0:
        batch_model_size = 4e9
        gpu_memory = max([
            torch.cuda.get_device_properties(i).total_memory
            for i in range(torch.cuda.device_count())
        ])
        gpus_per_trial = batch_model_size / gpu_memory

    # Setup storage path
    if ray_storage_path is None:
        ray_storage_path = os.path.realpath(".results")

    # Create tuner
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                run_training,
                data=data,
                task=task,
                col_stats_dict=col_stats_dict,
                use_ray=True,
            ),
            resources={"cpu": num_cpus, "gpu": gpus_per_trial}
        ),
        run_config=ray_train.RunConfig(
            name=ray_experiment_name,
            storage_path=ray_storage_path,
            callbacks=ray_callbacks,
            stop={"training_iteration": search_space["epochs"]},
            log_to_file=True,
        ),
        tune_config=tune.TuneConfig(
            metric=tune_metric_name,
            mode=metric_mode,
            scheduler=scheduler,
            num_samples=num_samples,
            trial_name_creator=lambda trial: f"{dataset_name}_{task_name}_{trial.trial_id}",
            trial_dirname_creator=lambda trial: trial.trial_id,
        ),
        param_space=search_space,
    )

    # Run tuning
    results = tuner.fit()

    # Print best results
    try:
        best_result = results.get_best_result(tune_metric_name, metric_mode)
        print("\n" + "="*80)
        print("BEST TRIAL RESULTS")
        print("="*80)
        print(f"Best trial config: {best_result.config}")
        print(f"Best trial metrics: {best_result.metrics}")
        print("="*80)
    except Exception as e:
        print(f"Error getting best result: {e}")

    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Simple GNN training test with optional Ray Tune")
    
    # Mode selection
    parser.add_argument("--mode", choices=["train", "tune"], default="train",
                       help="Run mode: train (single run) or tune (hyperparameter search)")
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Dataset and task
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--task", type=str, help="Task name")
    
    # Model config
    parser.add_argument("--model", choices=["sage", "dbformer", "sage_edge_attr"], 
                       default="sage", help="Model architecture")
    parser.add_argument("--tabular_model", choices=["resnet", "linear"], 
                       default="resnet", help="Tabular model type")
    parser.add_argument("--aggr", choices=["sum", "mean", "max"], 
                       default="sum", help="Aggregation function")
    parser.add_argument("--norm", choices=["batch_norm", "layer_norm"], 
                       default="batch_norm", help="Normalization type")
    parser.add_argument("--channels", type=int, default=64, help="Hidden channels")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--num_neighbors", type=int, default=16, help="Number of neighbors to sample")
    
    # Training config
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_steps_per_epoch", type=int, default=10, 
                       help="Max steps per epoch")
    
    # Data processing config
    parser.add_argument("--process_bridge", action="store_true", default=False, 
                       help="Process bridge tables")
    parser.add_argument("--bridge_strategy", 
                       choices=["default", "keep_attributes", "keep_table"], 
                       default="default")
    parser.add_argument("--process_hub", action="store_true", default=False, 
                       help="Process hub tables")
    parser.add_argument("--hub_strategy", 
                       choices=["default_combinations", "keep_attributes", "keep_table"], 
                       default="default_combinations")
    parser.add_argument("--cache_dir", type=str, default=".cache", 
                       help="Cache directory")
    
    # Ray Tune specific arguments
    parser.add_argument("--ray_address", type=str, default="local", 
                       help="Ray cluster address")
    parser.add_argument("--ray_storage", type=str, default=None, 
                       help="Ray storage path")
    parser.add_argument("--run_name", type=str, default=None, 
                       help="Experiment name")
    parser.add_argument("--mlflow_uri", type=str, default=None, 
                       help="MLflow tracking URI")
    parser.add_argument("--mlflow_experiment", type=str, default="gnn_experiment", 
                       help="MLflow experiment name")
    parser.add_argument("--aim_repo", type=str, default=None, 
                       help="Aim repo path")
    parser.add_argument("--num_samples", type=int, default=1, 
                       help="Number of trials for tuning")
    parser.add_argument("--num_gpus", type=float, default=0, 
                       help="Number of GPUs per trial")
    parser.add_argument("--num_cpus", type=int, default=1, 
                       help="Number of CPUs per trial")
    parser.add_argument("--tune_config", type=str, default=None,
                       help="Path to JSON file with Ray Tune search space overrides")

    args = parser.parse_args()

    # Default config (for standalone mode)
    config = {
        "dataset": "rel-f1",
        "task": "driver-position",
        "model": "sage_edge_attr",
        "tabular_model": "resnet",
        "aggr": "sum",
        "norm": "batch_norm",
        "channels": 128,
        "num_layers": 2,
        "num_neighbors": 16,
        "seed": 42,
        "lr": 0.001,
        "epochs": 20,
        "batch_size": 32,
        "max_steps_per_epoch": 10,
        "process_bridge": True,
        "bridge_strategy": "keep_attributes",
        "process_hub": True,
        "hub_strategy": "keep_table",
        "cache_dir": ".cache",
    }

    # Load config file if provided
    if args.config:
        with open(args.config) as f:
            file_config = json.load(f)
        config.update(file_config)

    # Override with CLI arguments (only if they differ from parser defaults)
    parser_defaults = {a.dest: a.default for a in parser._actions}
    for key, value in config.items():
        if hasattr(args, key):
            if getattr(args, key) == parser_defaults.get(key):
                setattr(args, key, value)

    # Validate required fields
    if not args.dataset or not args.task:
        parser.error("--dataset and --task are required (either via CLI or config file)")

    if args.mode == "train":
        # Standalone training mode
        print("="*80)
        print("STANDALONE TRAINING MODE")
        print("="*80)
        
        run_kwargs = {
            "dataset_name": args.dataset,
            "task_name": args.task,
            "model_architecture": args.model,
            "tabular_model": args.tabular_model,
            "random_seed": args.seed,
            "lr": args.lr,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "channels": args.channels,
            "num_layers": args.num_layers,
            "num_neighbors": args.num_neighbors,
            "max_steps_per_epoch": args.max_steps_per_epoch,
            "aggr": args.aggr,
            "norm": args.norm,
            "process_bridge": args.process_bridge,
            "bridge_strategy": args.bridge_strategy,
            "process_hub": args.process_hub,
            "hub_strategy": args.hub_strategy,
            "cache_dir": args.cache_dir,
        }
        print("Configuration:")
        for k, v in run_kwargs.items():
            print(f"  {k}: {v}")
        print()
        
        simple_train_test(**run_kwargs)
        
    else:
        # Ray Tune mode
        if not RAY_AVAILABLE:
            parser.error("Ray Tune is not installed. Install with: pip install ray[tune]")
        
        print("="*80)
        print("RAY TUNE HYPERPARAMETER SEARCH MODE")
        print("="*80)
        
        # Load tune config if provided
        tune_search_space = None
        if args.tune_config:
            with open(args.tune_config) as f:
                tune_search_space = json.load(f)
        
        print(f"Dataset: {args.dataset}")
        print(f"Task: {args.task}")
        print(f"Number of samples: {args.num_samples}")
        print(f"Resources per trial: {args.num_cpus} CPUs, {args.num_gpus} GPUs")
        print()
        
        run_ray_tuner(
            dataset_name=args.dataset,
            task_name=args.task,
            model_architecture=args.model,
            tabular_model=args.tabular_model,
            random_seed=args.seed,
            ray_address=args.ray_address,
            ray_storage_path=args.ray_storage,
            ray_experiment_name=args.run_name,
            mlflow_uri=args.mlflow_uri,
            mlflow_experiment=args.mlflow_experiment,
            aim_repo=args.aim_repo,
            num_samples=args.num_samples,
            num_gpus=args.num_gpus,
            num_cpus=args.num_cpus,
            cache_dir=args.cache_dir,
            process_bridge=args.process_bridge,
            bridge_strategy=args.bridge_strategy,
            process_hub=args.process_hub,
            hub_strategy=args.hub_strategy,
            tune_config=tune_search_space,
        )