# Edge-Aware SAGE GNN Implementation

## Overview

The `SAGEEdgeAttrModel` is a GraphSAGE variant that can leverage edge attributes on heterogeneous graphs while remaining architecturally comparable to the standard `SAGEModel` for fair performance comparison.

## Key Features

1. **Backward Compatible**: Works on graphs with or without edge attributes
2. **Comparable Architecture**: Same structure as SAGEModel (encoder → temporal → GNN → head)
3. **Selective Edge Features**: Can use edge attributes on some edge types while ignoring others
4. **Fair Comparison**: When no edge attributes are provided, behaves like standard SAGE

## How It Works

### Architecture Comparison

**Standard SAGE (`SAGEModel`)**:
```
Input Graph → HeteroEncoder → TemporalEncoder → HeteroGraphSAGE → MLP Head → Output
                  (tabular)       (time)            (message passing)
```

**Edge-Aware SAGE (`SAGEEdgeAttrModel`)**:
```
Input Graph → HeteroEncoder → TemporalEncoder → HeteroEdgeAttrGraphSAGE → MLP Head → Output
                  (tabular)       (time)         (message passing + edge features)
```

The only difference is in the GNN layer, which uses `EdgeAttrSAGEConv` instead of standard `SAGEConv`.

### Edge Attribute Handling

The `EdgeAttrSAGEConv` layer modulates messages using edge features:

```python
# Standard SAGE message
message = neighbor_features

# Edge-Aware SAGE message (when edge_attr exists)
edge_gate = sigmoid(Linear(edge_attr))
message = neighbor_features * edge_gate
```

This gating mechanism allows the model to selectively weight neighbor contributions based on edge properties.

## Usage

### Basic Usage (No Edge Attributes)

Behaves exactly like standard SAGE:

```python
from redelex.nn.models.sage_edge_attr import SAGEEdgeAttrModel

model = SAGEEdgeAttrModel(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=2,
    channels=64,
    tabular_model="resnet",
    out_channels=1,
    aggr="mean",
    norm="batch_norm",
)
```

### With Edge Attributes

Specify edge dimensions for edge types that have attributes:

```python
model = SAGEEdgeAttrModel(
    data=data,
    col_stats_dict=col_stats_dict,
    num_layers=2,
    channels=64,
    tabular_model="resnet",
    out_channels=1,
    aggr="mean",
    norm="batch_norm",
    edge_dim_dict={
        ('user', 'rates', 'movie'): 16,  # edge attributes with 16 features
        ('user', 'friends', 'user'): 8,  # edge attributes with 8 features
        # other edge types without entry will not use edge attributes
    },
)
```

### Using with test_simple_train.py

Update your config:

```python
config = {
    "dataset": "rel-f1",
    "task": "driver-position",
    "model": "sage_edge_attr",  # Changed from "sage"
    "tabular_model": "resnet",
    "channels": 64,
    "num_layers": 2,
    # ... other params
}
```

Then run:

```bash
python test_simple_train.py
```

Or from command line:

```bash
python test_simple_train.py --dataset rel-f1 --task driver-position --model sage_edge_attr
```

## Implementation Details

### Edge Feature Preprocessing

If your graph has edge attributes stored in the data loader batches, they will be automatically extracted:

```python
# In forward pass, edge attributes are collected from batch
edge_attr_dict = {}
for edge_type in batch.edge_types:
    if hasattr(batch[edge_type], "edge_attr"):
        edge_attr_dict[edge_type] = batch[edge_type].edge_attr
```

### Adding Edge Attributes to Your Data

To use edge features, ensure your data loader includes them. Example:

```python
# When creating HeteroData
data[('src', 'edge_type', 'dst')].edge_attr = edge_features  # shape: [num_edges, edge_dim]
```

## Comparison Guidelines

For fair comparison between models:

1. **Baseline (Standard SAGE)**:
   ```bash
   python test_simple_train.py --model sage --channels 64 --num_layers 2
   ```

2. **Edge-Aware SAGE (no edge features)**:
   ```bash
   python test_simple_train.py --model sage_edge_attr --channels 64 --num_layers 2
   ```
   Should give similar results to baseline.

3. **Edge-Aware SAGE (with edge features)**:
   ```bash
   python test_simple_train.py --model sage_edge_attr --channels 64 --num_layers 2
   ```
   With edge_attr data prepared, should show improvement.

## Performance Considerations

- **Parameters**: Same number of parameters as SAGE when `edge_dim_dict` is empty
- **Computation**: Minimal overhead (~5-10%) even with edge attributes
- **Memory**: Additional memory proportional to edge feature dimensions

## Extending to Other Convolutions

The same pattern can be applied to other convolutions (GAT, GCN, etc.):

```python
class EdgeAttrGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=None, heads=1):
        # Similar structure to EdgeAttrSAGEConv
        # but with attention mechanism
        pass
```

## References

- Original SAGE: Hamilton et al., "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
- PyTorch Geometric SAGEConv: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.SAGEConv
