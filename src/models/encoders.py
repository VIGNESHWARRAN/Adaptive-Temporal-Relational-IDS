"""
Fixed Temporal and Relational Encoders.
To isolate fusion method evaluation, the temporal encoder E_temp and relational encoder E_rel
remain frozen in capacity across all experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Union


class FixedTemporalEncoder(nn.Module):
    """
    Fixed Temporal Sequence Encoder (GRU-based).
    Input: Sequence matrix S_t of shape (Batch, K, input_dim)
    Output: Temporal embedding vector h_temp of shape (Batch, embed_dim)
    """

    def __init__(self, input_dim: int = 13, hidden_dim: int = 64, embed_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc_proj = nn.Linear(hidden_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq shape: (B, K, d_f)
        _, h_n = self.gru(x_seq)  # h_n shape: (num_layers, B, hidden_dim)
        h_last = h_n[-1]  # (B, hidden_dim)
        h_temp = self.layer_norm(F.relu(self.fc_proj(h_last)))
        return h_temp


class FixedRelationalEncoder(nn.Module):
    """
    Fixed Relational / Behavioral Graph Encoder.
    Processes graph snapshot node features X_v, edge indices E_t, and edge attributes X_e.
    Output: Relational summary embedding h_rel of shape (Batch, embed_dim)
    """

    def __init__(self, node_input_dim: int = 5, edge_input_dim: int = 13, embed_dim: int = 64):
        super().__init__()
        self.node_proj = nn.Linear(node_input_dim, 32)
        self.edge_proj = nn.Linear(edge_input_dim, 32)
        self.msg_mlp = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
        self.global_pool = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward_single_graph(self, x_v: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Single graph forward pass
        num_nodes = x_v.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            h_nodes = F.relu(self.node_proj(x_v))
            h_graph = torch.mean(h_nodes, dim=0, keepdim=True)
            return self.layer_norm(F.relu(self.global_pool(F.pad(h_graph, (0, h_graph.size(-1) - 32) if h_graph.size(-1) < 64 else (0, 0)))))

        src, dst = edge_index[0], edge_index[1]
        x_v_proj = self.node_proj(x_v)
        x_e_proj = self.edge_proj(edge_attr)

        # Message aggregation along edges
        msg_inputs = torch.cat([x_v_proj[src], x_e_proj], dim=-1)
        messages = self.msg_mlp(msg_inputs)

        # Aggregate messages at destination nodes
        node_agg = torch.zeros(num_nodes, messages.size(-1), device=x_v.device)
        node_agg.index_add_(0, dst, messages)

        # Global mean pooling over nodes
        h_graph = torch.mean(node_agg, dim=0, keepdim=True)
        h_rel = self.layer_norm(F.relu(self.global_pool(h_graph)))
        return h_rel

    def forward(self, batch_graphs: Union[List[Dict[str, torch.Tensor]], Any]) -> torch.Tensor:
        """
        Processes a list or batch of graph snapshots, returning shape (Batch, embed_dim)
        """
        embeddings = []
        if isinstance(batch_graphs, list):
            for g in batch_graphs:
                if hasattr(g, "x"):
                    xv, ei, ea = g.x, g.edge_index, g.edge_attr
                else:
                    xv, ei, ea = g["x"], g["edge_index"], g["edge_attr"]
                h_g = self.forward_single_graph(xv, ei, ea)
                embeddings.append(h_g)
            return torch.cat(embeddings, dim=0)
        elif hasattr(batch_graphs, "x"):
            # Single PyG Data object
            return self.forward_single_graph(batch_graphs.x, batch_graphs.edge_index, batch_graphs.edge_attr)
        else:
            raise TypeError("Unsupported graph batch type")
