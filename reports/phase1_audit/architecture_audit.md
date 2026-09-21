# 🏗️ MODEL ARCHITECTURE AUDIT REPORT

**Project**: Adaptive Temporal-Relational Intrusion Detection  
**Phase**: Phase 1 Controlled Experimentation Audit  
**Date**: September 17, 2026  

---

## 1. Architectural Blueprint & Target Flow

```text
Input Network Traffic (Selected Feature Vector X \in R^D)
                        │
                        ├─────────────────────────┐
                        ▼                         ▼
            Temporal Sub-vector           Relational Sub-vector
                        │                         │
                        ▼                         ▼
            FixedTemporalEncoder        FixedRelationalEncoder
            (2-Layer GRU, d_h=64)       (Projection & Aggregation, d_h=64)
                        │                         │
                        ▼                         ▼
                h_temp \in R^64           h_rel \in R^64
                        │                         │
                        └───────────┬─────────────┘
                                    ▼
                         Concatenation Fusion
                       [h_temp || h_rel] \in R^128
                                    │
                                    ▼
                           Linear(128 -> 64)
                                    │
                              BatchNorm1d
                                    │
                                  ReLU
                                    │
                              Fused R^64
                                    │
                                    ▼
                             Classifier Head
                         Linear(64 -> 32) -> ReLU
                         Dropout(p=0.2)
                         Linear(32 -> C)
                                    │
                                    ▼
                           Class Logits \in R^C
```

---

## 2. Component-by-Component Architectural Inspection

### A. Fixed Temporal Encoder (`FixedTemporalEncoder`)
- **Location**: `src/models/encoders.py` (Lines 12–62)
- **Structure**:
  - `nn.GRU`: Unidirectional, `num_layers = 2`, `hidden_dim = 64`, `batch_first = True`.
  - Projection: `nn.Linear(64, 64)` followed by `F.relu` and `nn.LayerNorm(64)`.
- **Functionality**: Takes 2D or 3D temporal input sequence, processes through 2 GRU layers, extracts final hidden state $h_n[-1]$, and projects to 64-dimensional temporal embedding $h_{\text{temp}}$.

### B. Fixed Relational Encoder (`FixedRelationalEncoder`)
- **Location**: `src/models/encoders.py` (Lines 64–177)
- **Structure**:
  - `node_proj`: `nn.Linear(node_in, 64)` followed by ReLU.
  - `message_mlp`: `nn.Sequential(Linear(64, 64) -> ReLU -> Linear(64, 64))`.
  - `global_pool`: Mean pooling followed by `nn.Linear(64, 64)` and `nn.LayerNorm(64)`.
- **Functionality**: Accepts graph snapshot objects or batched 2D feature representations, applies node projection and message passing, pools globally, and yields 64-dimensional relational embedding $h_{\text{rel}}$.

### C. Multimodal Fusion Module (`ConcatenationFusion`)
- **Location**: `src/models/fusion.py` (Lines 21–65)
- **Structure**:
  - Input: Concatenation of $h_{\text{temp}} \in \mathbb{R}^{64}$ and $h_{\text{rel}} \in \mathbb{R}^{64} \implies \mathbb{R}^{128}$.
  - Projection: `nn.Linear(128, 64)`.
  - Normalization & Activation: `nn.BatchNorm1d(64)` followed by `nn.ReLU()`.
- **Functionality**: Early baseline concatenation fusion producing a unified 64-dimensional multimodal state representation.

### D. Classifier Head (`MultimodalFusionClassifier`)
- **Location**: `src/models/classifiers.py` (Lines 60–65)
- **Structure**:
  - `Linear(64, 32)`
  - `ReLU()`
  - `Dropout(p=0.2)`
  - `Linear(32, C)` where $C$ is the number of target classes ($C=2$ for NetFlow datasets, $C=15$ or $11$ for CSE-CIC).

---

## 3. Parameter Counts & Dimension Audit

| Sub-module | Input Shape | Output Shape | Layer Details | Parameter Count (Approx) |
| :--- | :--- | :--- | :--- | :--- |
| **Temporal GRU** | $(B, 1, D)$ | $(B, 64)$ | 2-layer GRU ($d_h=64$) + Linear(64, 64) + LayerNorm | ~50,000 |
| **Relational GNN** | $(B, D)$ | $(B, 64)$ | Linear(D, 64) + MLP(64, 64) + Pool + LayerNorm | ~12,000 |
| **Concat Fusion** | $(B, 64), (B, 64)$ | $(B, 64)$ | Linear(128, 64) + BatchNorm1d(64) + ReLU | 8,320 |
| **Classifier Head** | $(B, 64)$ | $(B, C)$ | Linear(64, 32) + ReLU + Dropout + Linear(32, C) | 2,144 + $33 \times C$ |

---

## 4. Control Across Experiments Verification

- **Architectural Fixedness**: The exact model instance `MultimodalFusionClassifier(fusion_type="concat", dropout_rate=0.2)` is instantiated identically in `scripts/run_phase1_experiments.py` across all 9 experiments.
- **Hyperparameter Stability**: Hidden dimensions ($64$), layer counts ($2$), dropout ($0.2$), and learning rate ($0.001$) remain strictly unchanged across standard CE, weighted CE, and focal loss runs.
- **Audit Conclusion**: Architectural control is **VERIFIED** and completely isolated from loss function modifications.
