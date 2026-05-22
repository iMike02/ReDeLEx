# README: Running `dbgnn_experiment.py`

This document explains how to run:

- `experiments/original/dbgnn_experiment.py`

It also documents the custom parts of this ReDeLEx pipeline:

1. `get_data_custom` (custom data preparation logic)
2. The modified `GraphSAGEEdgeAttr` model (`SAGEEdgeAttrModel`)

## 1) What this script does

`dbgnn_experiment.py` trains a relational GNN pipeline on RelBench-style tasks.

Main features:

- Supports models: `sage`, `dbformer`, `sage_edge_attr`
- Builds one graph once per run configuration, then trains across multiple seeds
- Supports bridge/hub processing switches in graph construction
- Logs per-seed and aggregated results (JSON + CSV)
- Supports optional linear learning-rate decay

---

## 2) How to run

Run from the ReDeLEx project root:

```bash
cd /ReDeLEx
python experiments/original/dbgnn_experiment.py
```

The script already contains defaults in its internal `config` dictionary, so the command above runs immediately.

### Common custom run

```bash
python experiments/original/dbgnn_experiment.py \
  --dataset rel-f1 \
  --task driver-position \
  --model sage_edge_attr \
  --tabular_model resnet \
  --seeds 42 43 44 45 46 \
  --batch_size 1024 \
  --num_layers 2 \
  --num_neighbors 32 \
  --process_bridge \
  --bridge_strategy keep_attributes \
  --process_hub \
  --hub_strategy keep_table
```

### Core CLI arguments

- `--dataset`, `--task`: task selection
- `--model`: `sage | dbformer | sage_edge_attr`
- `--tabular_model`: `resnet | linear`
- `--seeds`: list of seeds for repeated runs
- `--process_bridge`: enable bridge-table processing
- `--bridge_strategy`: `default | keep_attributes | keep_table`
- `--process_hub`: enable hub-table processing
- `--hub_strategy`: `default_combinations | keep_attributes | keep_table`
- `--toggle_logging` / `--no_toggle_logging`
- `--toggle_summary_csv` / `--no_toggle_summary_csv`

### Output locations

By default (`--log_dir logs/training_logs`), outputs are written under:

- `logs/training_logs/<dataset>_<task>_<model>/configs.json` (sweep manifest)
- `logs/training_logs/<dataset>_<task>_<model>/cfg_<k>.json` (seed-wise history + aggregate)
- `logs/training_logs/<dataset>_<task>_<model>/cfg_<k>.csv` (aggregated summary row)

`cfg_<k>` is determined from the bridge/hub strategy combination.

---

## 3) `get_data_custom` contribution

Defined in:

- `experiments/utils.py` (`get_data_custom`)

### What it is

`get_data_custom(...)` is a customized data-building entry point that extends the standard graph materialization path.

It does the following:

1. Loads dataset/task and obtains DB snapshot (or task-modified DB)
2. Builds attribute schema and text embedder config
3. Calls `make_pkey_fkey_graph_custom(...)` instead of the default graph maker
4. Applies bridge/hub processing logic via:
   - `process_bridge`, `bridgeStrategy`
   - `process_hub`, `hubStrategy`
5. Returns `(task, data, col_stats_dict)` for model training

### Where it is used

In this script, usage is centralized in:

- `build_training_data(...)` inside `experiments/original/dbgnn_experiment.py`

Flow:

- CLI args (`--process_bridge`, `--bridge_strategy`, `--process_hub`, `--hub_strategy`)
- passed to `build_training_data(...)`
- forwarded to `get_data_custom(...)`
- graph + stats returned and reused across all seeds for that config

So `get_data_custom` is the key point where bridge/hub graph construction choices enter the training pipeline.

---

## 4) Modified GraphSAGEEdgeAttr model

Implemented in:

- `redelex/nn/models/sage_edge_attr.py`

Model class used by this script:

- `SAGEEdgeAttrModel`

### How the script selects it

In `dbgnn_experiment.py`:

- `get_model(...)` returns `SAGEEdgeAttrModel(...)` when `--model sage_edge_attr`

### What is modified vs standard GraphSAGE

The edge-aware path introduces custom message passing:

- `EdgeAttrSAGEConv` extends `MessagePassing`
- If edge attributes are present, message is:
  - `message = x_j + Linear(edge_attr)`
- If edge attributes are absent, fallback is standard behavior:
  - `message = x_j`

Then normal GraphSAGE-style transformations are applied (`lin_l`, `lin_r`, optional normalization).

### Heterogeneous edge-aware stack

- `HeteroEdgeAttrGraphSAGE` creates one `EdgeAttrSAGEConv` per edge type
- Supports mixed edge types: some with edge attributes, some without
- Uses `HeteroConv(..., aggr="sum")` + per-node-type `LayerNorm` + ReLU for each layer

### Automatic edge dimension discovery

`SAGEEdgeAttrModel` can infer `edge_dim_dict` automatically from `data[edge_type].edge_attr` shape via `_derive_edge_dim_dict(...)`.

That means the model works without manually specifying edge dimensions in common cases.

### End-to-end architecture

`SAGEEdgeAttrModel` preserves the same high-level structure as baseline SAGE for fair comparison:

1. `HeteroEncoder` for tabular node features
2. `HeteroTemporalEncoder` for time-aware tasks
3. `HeteroEdgeAttrGraphSAGE` for message passing with optional edge attributes
4. `MLP` prediction head

This modification is focused on the GNN message-passing core while keeping the rest of the training stack comparable.

---

## 5) Practical notes

- Run from ReDeLEx root so imports like `from experiments.utils ...` resolve correctly.
- If a bridge/hub strategy combination is passed that is not in the canonical config mapping, config-id resolution may fail.
- For faster sanity checks, reduce:
  - `--max_steps_per_epoch`
  - `--min_total_steps`
  - `--seeds`

Example quick debug run:

```bash
python experiments/original/dbgnn_experiment.py \
  --dataset rel-f1 \
  --task driver-position \
  --model sage_edge_attr \
  --seeds 42 \
  --max_steps_per_epoch 50 \
  --min_total_steps 200 \
  --batch_size 256
```
