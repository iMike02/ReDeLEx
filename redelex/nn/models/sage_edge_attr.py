"""
SAGE-based GNN that supports edge attributes on heterogeneous graphs.
This model is designed to be comparable to the standard SAGEModel while
being able to leverage edge features when available.
"""

from typing import Any, Dict, List, Literal, Optional

import torch
from torch import Tensor

from torch_frame.data.stats import StatType
from torch_frame.nn import ResNet

from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP, HeteroConv, LayerNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import NodeType, EdgeType
from torch_geometric.utils import softmax

from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder

from redelex.nn.encoders import LinearRowEncoder


class EdgeAttrSAGEConv(MessagePassing):
    """
    SAGEConv variant that incorporates edge attributes.
    
    When edge attributes are present, they are used to modulate the message passing.
    When edge attributes are absent, it behaves like standard SAGEConv.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        aggr: str = "mean",
        normalize: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.normalize = normalize
        
        # Standard SAGE transformation
        self.lin_l = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=False)
        
        # Edge attribute transformation (if edge features are present)
        # in_channels because edge_attr is transformed before being added to the message, so it needs to match the message dimension
        if edge_dim is not None:
            self.lin_edge = torch.nn.Linear(edge_dim, in_channels, bias=False)
        else:
            self.lin_edge = None
            
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        # x can be Tensor [N, in_channels] or tuple (x_src, x_dst) for bipartite graphs
        # edge_attr has shape [E, edge_dim] if provided
        
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.lin_l(out)
        
        # Extract destination nodes (for bipartite: x[1], otherwise: x) (x_dst ~ x_r)
        x_dst = x[1] if isinstance(x, tuple) else x

        # Add self-connection
        if x_dst.size(-1) == self.in_channels:
            out = out + self.lin_r(x_dst)
        
        if self.normalize:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
            
        return out
    
    def message(self, x_j: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        # x_j has shape [E, in_channels] (source node features)
        # edge_attr has shape [E, edge_dim] if provided
        
        if edge_attr is not None and self.lin_edge is not None:
            # Incorporate edge features: Add information from edge attributes to the message
            edge_features = self.lin_edge(edge_attr)
            return x_j + edge_features
        else:
            return x_j


class HeteroEdgeAttrGraphSAGE(torch.nn.Module):
    """
    Heterogeneous Graph SAGE that can optionally use edge attributes.
    
    This is comparable to HeteroGraphSAGE but supports edge attributes
    for edges that have them, while still working for edges without attributes.
    """
    
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        edge_dim_dict: Optional[Dict[EdgeType, int]] = None,
        aggr: str = "mean",
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.edge_dim_dict = edge_dim_dict or {}
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                edge_dim = self.edge_dim_dict.get(edge_type, None)
                conv_dict[edge_type] = EdgeAttrSAGEConv(
                    channels,
                    channels,
                    edge_dim=edge_dim,
                    aggr=aggr,
                )
            conv = HeteroConv(conv_dict, aggr="sum")
            self.convs.append(conv)
        
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()
    
    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None,
        num_sampled_nodes_dict: Optional[Dict[NodeType, List[int]]] = None,
        num_sampled_edges_dict: Optional[Dict[EdgeType, List[int]]] = None,
    ) -> Dict[NodeType, Tensor]:
        
        edge_attr_dict = edge_attr_dict or {}
        
        for conv, norm_dict in zip(self.convs, self.norms):
            # HeteroConv handles x_dict → (x_src, x_dst) extraction internally
            # Pass edge_attr_dict (empty dict if no edge attributes)
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        return x_dict


class SAGEEdgeAttrModel(torch.nn.Module):
    """
    SAGE model that supports edge attributes while remaining comparable to SAGEModel.
    
    Usage:
        model = SAGEEdgeAttrModel(
            data=data,
            col_stats_dict=col_stats_dict,
            num_layers=2,
            channels=64,
            tabular_model="resnet",
            out_channels=1,
            aggr="mean",
            norm="batch_norm",
            edge_dim_dict={('table1', 'rel', 'table2'): 16},  # optional
        )
    """
    
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        tabular_model: Literal["resnet", "linear"],
        out_channels: int,
        aggr: str,
        norm: str,
        edge_dim_dict: Optional[Dict[EdgeType, int]] = None,
    ):
        super().__init__()

        if edge_dim_dict is None:
            edge_dim_dict = self._derive_edge_dim_dict(data)
        
        def get_tabular_model(tabular_model: str):
            if tabular_model == "resnet":
                return ResNet, {
                    "channels": 128,
                    "num_layers": 4,
                }
            elif tabular_model == "linear":
                return LinearRowEncoder, {"channels": 128}
            else:
                raise ValueError(f"Unknown tabular_model: {tabular_model}")
        
        encoder_cls, encoder_kwargs = get_tabular_model(tabular_model)
        
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
            torch_frame_model_cls=encoder_cls,
            torch_frame_model_kwargs=encoder_kwargs,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroEdgeAttrGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            edge_dim_dict=edge_dim_dict,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )
        
        self.reset_parameters()

    @staticmethod
    def _derive_edge_dim_dict(data: HeteroData) -> Dict[EdgeType, int]:
        edge_dim_dict: Dict[EdgeType, int] = {}
        for edge_type in data.edge_types:
            edge_attr = getattr(data[edge_type], "edge_attr", None)
            if torch.is_tensor(edge_attr):
                tensor_edge_attr = edge_attr
                if tensor_edge_attr.dim() >= 2:
                    edge_dim_dict[edge_type] = int(tensor_edge_attr.size(-1))
        return edge_dim_dict
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
    
    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        x_dict = self.encoder(batch.tf_dict)
        
        if hasattr(batch[entity_table], "seed_time"):
            seed_time = batch[entity_table].seed_time
            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )
            
            for node_type, rel_time in rel_time_dict.items():
                x_dict[node_type] = x_dict[node_type] + rel_time
        
        # Extract edge attributes if they exist in the batch
        edge_attr_dict = {}
        for edge_type in batch.edge_types:
            if hasattr(batch[edge_type], "edge_attr"):
                edge_attr_dict[edge_type] = batch[edge_type].edge_attr
        
        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
            edge_attr_dict=edge_attr_dict if edge_attr_dict else None,
            num_sampled_nodes_dict=batch.num_sampled_nodes_dict,
            num_sampled_edges_dict=batch.num_sampled_edges_dict,
        )
        
        if hasattr(batch[entity_table], "seed_time"):
            return self.head(x_dict[entity_table][: seed_time.size(0)])
        
        return self.head(x_dict[entity_table])
