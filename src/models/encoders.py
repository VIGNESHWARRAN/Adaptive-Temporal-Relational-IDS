"""Fixed Temporal and Relational Encoders for Intrusion Detection.

Architectures are kept strictly frozen across all Phase 1 experiments.
"""

import torch
import torch.nn as nn


class FixedTemporalEncoder(nn.Module):
    """Fixed GRU-based Temporal Sequence Encoder.
    
    Architecture:
    - 2-layer Unidirectional GRU with hidden dimension 64.
    - Linear projection from GRU output (64) -> 64.
    - ReLU activation.
    - LayerNorm (64).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Temporal sequence tensor of shape (batch_size, seq_len, input_dim)
               or (batch_size, input_dim) if unsequenced.
        
        Returns:
            Temporal embedding tensor of shape (batch_size, 64)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        gru_out, h_n = self.gru(x)  # gru_out: (batch_size, seq_len, 64)
        # Take the last time step embedding
        last_step = gru_out[:, -1, :]  # (batch_size, 64)
        
        out = self.proj(last_step)
        out = self.relu(out)
        out = self.layer_norm(out)
        return out


class FixedRelationalEncoder(nn.Module):
    """Fixed Relational Graph Encoder.
    
    Structure:
    - Node feature projection (node_dim -> 64)
    - Edge feature projection (edge_dim -> 64)
    - Message passing MLP aggregation (64 -> 64)
    - Destination-node aggregation / Global pooling
    - Linear projection -> 64
    - LayerNorm (64)
    """

    def __init__(self, node_dim: int, edge_dim: int = 0, hidden_dim: int = 64):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        self.node_proj = nn.Linear(node_dim, hidden_dim)
        if edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_proj = None

        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * (2 if edge_dim > 0 else 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.final_proj = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_feats: torch.Tensor, edge_feats: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            node_feats: Tensor of shape (batch_size, node_dim)
            edge_feats: Optional tensor of shape (batch_size, edge_dim)
            
        Returns:
            Relational embedding tensor of shape (batch_size, 64)
        """
        h_node = self.relu(self.node_proj(node_feats))
        
        if self.edge_proj is not None and edge_feats is not None:
            h_edge = self.relu(self.edge_proj(edge_feats))
            combined = torch.cat([h_node, h_edge], dim=-1)
        else:
            combined = h_node

        msg = self.message_mlp(combined)
        out = self.final_proj(msg)
        out = self.relu(out)
        out = self.layer_norm(out)
        return out
