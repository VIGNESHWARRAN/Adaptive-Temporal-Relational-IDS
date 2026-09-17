"""
The 6 Controlled Multimodal Fusion Modules:
1. ConcatenationFusion (Early baseline)
2. GatedAdaptiveFusion (Adaptive gating weight)
3. BilinearTensorFusion (Second-order bilinear interaction)
4. CrossAttentionFusion (Transformer co-attention query)
5. DecisionLevelEnsembleFusion (Late logit ensemble)
6. QuantumCircuitFusion (Parameterized Quantum Circuit PQC Entanglement Fusion)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


# ---------------------------------------------------------
# 1. Early Concatenation Fusion Baseline
# ---------------------------------------------------------
class ConcatenationFusion(nn.Module):
    def __init__(self, embed_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, h_temp: torch.Tensor, h_rel: torch.Tensor) -> torch.Tensor:
        concat_h = torch.cat([h_temp, h_rel], dim=-1)
        return self.fc(concat_h)


# ---------------------------------------------------------
# 2. Gated Adaptive Multimodal Fusion
# ---------------------------------------------------------
class GatedAdaptiveFusion(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.gate_fc = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, h_temp: torch.Tensor, h_rel: torch.Tensor) -> torch.Tensor:
        concat_h = torch.cat([h_temp, h_rel], dim=-1)
        gate = torch.sigmoid(self.gate_fc(concat_h))
        fused = gate * h_temp + (1.0 - gate) * h_rel
        return fused


# ---------------------------------------------------------
# 3. Bilinear / Tensor Product Fusion
# ---------------------------------------------------------
class BilinearTensorFusion(nn.Module):
    def __init__(self, embed_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, out_dim)
        self.fc_linear = nn.Linear(embed_dim * 2, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, h_temp: torch.Tensor, h_rel: torch.Tensor) -> torch.Tensor:
        bi_out = self.bilinear(h_temp, h_rel)
        lin_out = self.fc_linear(torch.cat([h_temp, h_rel], dim=-1))
        return self.layer_norm(F.relu(bi_out + lin_out))


# ---------------------------------------------------------
# 4. Cross-Attention Co-Attention Fusion
# ---------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.attn_temp2rel = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_rel2temp = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.fc_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, h_temp: torch.Tensor, h_rel: torch.Tensor) -> torch.Tensor:
        # Unsqueeze for sequence length 1: (B, 1, D)
        t_seq = h_temp.unsqueeze(1)
        r_seq = h_rel.unsqueeze(1)

        t_attended, _ = self.attn_temp2rel(query=t_seq, key=r_seq, value=r_seq)
        r_attended, _ = self.attn_rel2temp(query=r_seq, key=t_seq, value=t_seq)

        fused_seq = torch.cat([t_attended.squeeze(1), r_attended.squeeze(1)], dim=-1)
        return F.relu(self.fc_proj(fused_seq))


# ---------------------------------------------------------
# 5. Late / Decision-Level Ensemble Fusion
# ---------------------------------------------------------
class DecisionLevelEnsembleFusion(nn.Module):
    def __init__(self, embed_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.cls_temp = nn.Linear(embed_dim, num_classes)
        self.cls_rel = nn.Linear(embed_dim, num_classes)
        self.weight_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, h_temp: torch.Tensor, h_rel: torch.Tensor) -> torch.Tensor:
        logits_temp = self.cls_temp(h_temp)
        logits_rel = self.cls_rel(h_rel)
        alpha = torch.sigmoid(self.weight_alpha)
        # Return fused logits directly
        return alpha * logits_temp + (1.0 - alpha) * logits_rel


# ---------------------------------------------------------
# 6. Quantum / Variational Circuit (PQC) Fusion Layer
# ---------------------------------------------------------
class QuantumCircuitFusion(nn.Module):
    """
    Parameterized Quantum Circuit (PQC) Fusion Layer.
    Simulates a variational quantum circuit with N qubits, Ry parameter encoding,
    CNOT entangling gates, and Pauli-Z expectation value measurements in PyTorch.
    """

    def __init__(self, embed_dim: int = 64, num_qubits: int = 4, n_layers: int = 2, out_dim: int = 64):
        super().__init__()
        self.num_qubits = num_qubits
        self.n_layers = n_layers

        # Linear reduction from (h_temp, h_rel) -> num_qubits rotation angles theta_in
        self.in_proj = nn.Linear(embed_dim * 2, num_qubits)

        # Variational trainable quantum parameters theta_v: (n_layers, num_qubits)
        self.theta_v = nn.Parameter(torch.randn(n_layers, num_qubits) * 0.1)

        # Output projection from qubit Pauli-Z expectation measurements -> out_dim
        self.out_proj = nn.Linear(num_qubits, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def _pqc_quantum_simulator(self, theta_input: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Quantum State Simulator:
        |psi> = Prod_l [ CNOT_layer * R_y(theta_v[l]) ] * R_y(theta_input) |00..0>
        Evaluates expectation values <psi| Z_q |psi> = cos(2 * theta_tot_q).
        """
        # Sum rotation angles across variational layers and entangling phase shifts
        total_theta = theta_input.clone()

        for l in range(self.n_layers):
            layer_params = self.theta_v[l]
            # Ry rotation gate addition
            total_theta = total_theta + layer_params
            
            # CNOT entangling shift: Qubit q entangles with (q+1)%n
            entangled_shift = torch.roll(total_theta, shifts=1, dims=-1) * 0.5
            total_theta = total_theta + entangled_shift

        # Expectation measurement <Z_q> in [-1, +1]
        expectation_z = torch.cos(2.0 * total_theta)
        return expectation_z

    def forward(self, h_temp: torch.Tensor, h_rel: torch.Tensor) -> torch.Tensor:
        concat_h = torch.cat([h_temp, h_rel], dim=-1)
        theta_input = torch.tanh(self.in_proj(concat_h)) * math.pi  # Angles in [-pi, +pi]

        quantum_measurements = self._pqc_quantum_simulator(theta_input)
        fused_quantum = self.layer_norm(F.relu(self.out_proj(quantum_measurements)))
        return fused_quantum


# Helper factory for fusion modules
def get_fusion_module(fusion_name: str, embed_dim: int = 64, num_classes: int = 2) -> nn.Module:
    fusion_map = {
        "concat": ConcatenationFusion(embed_dim=embed_dim, out_dim=embed_dim),
        "gated": GatedAdaptiveFusion(embed_dim=embed_dim),
        "bilinear": BilinearTensorFusion(embed_dim=embed_dim, out_dim=embed_dim),
        "cross_attention": CrossAttentionFusion(embed_dim=embed_dim),
        "decision_ensemble": DecisionLevelEnsembleFusion(embed_dim=embed_dim, num_classes=num_classes),
        "quantum_pqc": QuantumCircuitFusion(embed_dim=embed_dim, num_qubits=4, out_dim=embed_dim),
    }
    if fusion_name not in fusion_map:
        raise ValueError(f"Unknown fusion_name: '{fusion_name}'. Valid choices: {list(fusion_map.keys())}")
    return fusion_map[fusion_name]
