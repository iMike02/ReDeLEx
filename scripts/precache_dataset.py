from argparse import ArgumentParser
from pathlib import Path

from experiments.original.dbgnn_train import build_strategy_run_configs
from experiments.utils import get_cache_path, get_data_custom


def main() -> None:
    parser = ArgumentParser(
        description="Precache dataset artifacts used by dbgnn training."
    )
    parser.add_argument("--dataset", type=str, default="ctu-adventureworks")
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--run_all_configs", action="store_true", default=True)
    parser.add_argument(
        "--no_run_all_configs", action="store_false", dest="run_all_configs"
    )
    parser.add_argument("--process_bridge", action="store_true", default=False)
    parser.add_argument("--bridge_strategy", type=str, default="default")
    parser.add_argument("--process_hub", action="store_true", default=False)
    parser.add_argument("--hub_strategy", type=str, default="default_combinations")
    parser.add_argument(
        "--bridge_strategy_options",
        nargs="+",
        default=["default", "keep_attributes", "keep_table"],
    )
    parser.add_argument(
        "--hub_strategy_options",
        nargs="+",
        default=["default_combinations", "keep_attributes", "keep_table"],
    )
    parser.add_argument("--start_config", type=int, default=1)
    parser.add_argument("--end_config", type=int, default=None)
    args = parser.parse_args()

    if args.start_config < 1:
        raise ValueError("--start_config must be >= 1")
    if args.end_config is not None and args.end_config < args.start_config:
        raise ValueError("--end_config must be >= --start_config")

    cache_path = get_cache_path(args.dataset, args.task, args.cache_dir)
    Path(cache_path).mkdir(parents=True, exist_ok=True)

    run_configs = build_strategy_run_configs(
        run_all_configs=args.run_all_configs,
        process_bridge=args.process_bridge,
        bridge_strategy=args.bridge_strategy,
        process_hub=args.process_hub,
        hub_strategy=args.hub_strategy,
        bridge_strategy_options=args.bridge_strategy_options,
        hub_strategy_options=args.hub_strategy_options,
    )

    print(f"Pre-caching dataset={args.dataset} task={args.task}")
    print(f"Cache path: {cache_path}")
    print(f"Total config combinations: {len(run_configs)}")

    for config_index, run_cfg in enumerate(run_configs, start=1):
        if config_index < args.start_config:
            continue
        if args.end_config is not None and config_index > args.end_config:
            break

        process_bridge = bool(run_cfg["process_bridge"])
        bridge_strategy = str(run_cfg["bridge_strategy"])
        process_hub = bool(run_cfg["process_hub"])
        hub_strategy = str(run_cfg["hub_strategy"])

        print(
            f"[cfg_{config_index}] process_bridge={process_bridge} bridge_strategy={bridge_strategy} "
            f"process_hub={process_hub} hub_strategy={hub_strategy}"
        )

        # This call materializes graph/text features into cache for this config combination.
        get_data_custom(
            dataset_name=args.dataset,
            task_name=args.task,
            cache_path=str(cache_path),
            process_bridge=process_bridge,
            bridgeStrategy=bridge_strategy,
            process_hub=process_hub,
            hubStrategy=hub_strategy,
        )

    print("Pre-cache completed.")


if __name__ == "__main__":
    main()
