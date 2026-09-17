"""Concatenation Fusion module for Phase 1 experimentation.

Kept strictly fixed across all Phase 1 experiments with fusion_type = "concat".
"""

import torch
import torch.nn as nn


class ConcatenationFusion(nn.Module):
    """Concatenation Multimodal Fusion Module.
    
    Concatenates Temporal Embedding (64) and Relational Embedding (64) -> (128)
    Linear (128 -> 64) -> BatchNorm1d (64) -> ReLU -> Fused Representation (64).
    """

    def __init__(self, temp_dim: int = 64, rel_dim: int = 64, fused_dim: int = 64):
        super().__init__()
        self.temp_dim = temp_dim
        self.rel_dim = rel_dim
        self.fused_dim = fused_dim

        in_dim = temp_dim + rel_dim
        self.linear = nn.Linear(in_dim, fused_dim)
        self.bn = nn.BatchNorm1d(fused_dim)
        self.relu = nn.ReLU()

    def forward(self, temp_emb: torch.Tensor, rel_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            temp_emb: Temporal embedding tensor of shape (batch_size, 64)
            rel_emb: Relational embedding tensor of shape (batch_size, 64)
            
        Returns:
            Fused embedding tensor of shape (batch_size, 64)
        """
        concat = torch.cat([temp_emb, rel_emb], dim=-1)  # (batch_size, 128)
        out = self.linear(concat)                        # (batch_size, 64)
        if out.size(0) > 1:
            out = self.bn(out)                           # BatchNorm requires batch_size > 1
        out = self.relu(out)
        return out
