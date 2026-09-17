"""Fixed Temporal and Relational Encoders for Adaptive Intrusion Detection.

Architectures are kept strictly frozen across all experiments to isolate fusion and loss function evaluations.
"""

from typing import List, Dict, Any, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedTemporalEncoder(nn.Module):
    """Fixed GRU-based Temporal Sequence Encoder.

    Architecture:
    - 2-layer Unidirectional GRU with hidden dimension 64.
    - Linear projection from GRU output (64) -> embed_dim (64).
    - ReLU activation.
    - LayerNorm (64).
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 64,
        embed_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc_proj = nn.Linear(hidden_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Temporal sequence tensor of shape (batch_size, seq_len, input_dim)
               or (batch_size, input_dim) if unsequenced.

        Returns:
            Temporal embedding tensor of shape (batch_size, embed_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        _, h_n = self.gru(x)     # h_n shape: (num_layers, batch_size, hidden_dim)
        h_last = h_n[-1]         # (batch_size, hidden_dim)
        out = self.layer_norm(F.relu(self.fc_proj(h_last)))
        return out


class FixedRelationalEncoder(nn.Module):
    """Fixed Relational / Behavioral Graph Encoder.

    Supports:
    1. Direct graph snapshot inputs (PyG Data objects or graph feature dicts).
    2. Batched node and edge feature tensors (batch_size, node_dim) / (batch_size, edge_dim).
    """

    def __init__(
        self,
        node_input_dim: int = 5,
        edge_input_dim: int = 0,
        embed_dim: int = 64,
        node_dim: Optional[int] = None,
        edge_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        node_in = node_dim if node_dim is not None else node_input_dim
        edge_in = edge_dim if edge_dim is not None else edge_input_dim
        out_dim = hidden_dim if hidden_dim is not None else embed_dim

        self.node_input_dim = node_in
        self.edge_input_dim = edge_in
        self.embed_dim = out_dim

        self.node_proj = nn.Linear(node_in, 32 if edge_in > 0 else out_dim)
        if edge_in > 0:
            self.edge_proj = nn.Linear(edge_in, 32)
            self.message_mlp = nn.Sequential(
                nn.Linear(32 + 32, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim),
            )
        else:
            self.edge_proj = None
            self.message_mlp = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )

        self.global_pool = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward_single_graph(
        self, x_v: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:

        num_nodes = x_v.size(0)
        num_edges = edge_index.size(1) if edge_index is not None and edge_index.dim() > 1 else 0

        if num_edges == 0 or self.edge_proj is None or edge_attr is None:
            h_nodes = F.relu(self.node_proj(x_v))
            h_graph = torch.mean(h_nodes, dim=0, keepdim=True)
            if h_graph.size(-1) < self.embed_dim:
                h_graph = F.pad(h_graph, (0, self.embed_dim - h_graph.size(-1)))
            return self.layer_norm(F.relu(self.global_pool(h_graph)))

        src, dst = edge_index[0], edge_index[1]
        x_v_proj = self.node_proj(x_v)
        x_e_proj = self.edge_proj(edge_attr)

        msg_inputs = torch.cat([x_v_proj[src], x_e_proj], dim=-1)
        messages = self.message_mlp(msg_inputs)

        node_agg = torch.zeros(num_nodes, messages.size(-1), device=x_v.device)
        node_agg.index_add_(0, dst, messages)

        h_graph = torch.mean(node_agg, dim=0, keepdim=True)
        h_rel = self.layer_norm(F.relu(self.global_pool(h_graph)))
        return h_rel

    def forward(
        self,
        node_feats: Union[torch.Tensor, List[Dict[str, torch.Tensor]], Any],
        edge_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Accepts either:
        1. (batch_graphs): A list/batch of PyTorch Geometric / Graph objects.
        2. (node_feats, edge_feats): Batched 2D feature tensors.
        """
        # Case 1: List of PyG Data objects or graph dicts
        if isinstance(node_feats, list):
            embeddings = []
            for g in node_feats:
                if hasattr(g, "x"):
                    xv, ei, ea = g.x, g.edge_index, g.edge_attr
                else:
                    xv, ei, ea = g["x"], g["edge_index"], g["edge_attr"]
                embeddings.append(self.forward_single_graph(xv, ei, ea))
            return torch.cat(embeddings, dim=0)

        # Single PyG Data object
        if hasattr(node_feats, "x"):
            return self.forward_single_graph(node_feats.x, node_feats.edge_index, node_feats.edge_attr)

        # Case 2: 2D Tensor inputs (batch_size, node_dim)
        if isinstance(node_feats, torch.Tensor):
            h_node = F.relu(self.node_proj(node_feats))

            if self.edge_proj is not None and edge_feats is not None:
                h_edge = F.relu(self.edge_proj(edge_feats))
                combined = torch.cat([h_node, h_edge], dim=-1)
                msg = self.message_mlp(combined)
            else:
                msg = self.message_mlp(h_node)

            out = self.layer_norm(F.relu(self.global_pool(msg)))
            return out

        raise TypeError(f"Unsupported input type for RelationalEncoder: {type(node_feats)}")
