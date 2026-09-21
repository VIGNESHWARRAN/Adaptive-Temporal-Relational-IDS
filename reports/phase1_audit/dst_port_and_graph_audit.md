# 🔌 DST PORT & GRAPH NODE IDENTIFIER AUDIT REPORT

**Project**: Adaptive Temporal-Relational Intrusion Detection for Evolving Network Traffic  
**Phase**: Phase 1 Pre-Correction Validation & Feature/Graph Audit  
**Date**: September 17, 2026  

---

## 1. Executive Summary

This report audits:
1. The exact usage and potential shortcut leakage of `Dst Port` (Destination Port) across tabular, temporal, and relational feature modules.
2. The construction of graph node identifiers in `FixedRelationalEncoder` to determine whether transductive graph leakage or test-set node information influences training representations.

---

## 2. Usage & Data Flow of `Dst Port`

### A. Location in Feature Selection Maps
In `configs/feature_selection/cse_cic_ids2018_features.json`:
- `selected_features`: Includes `"Dst Port"`
- `relational_features`: `["Dst Port"]`
- `graph_edge_features`: `["Dst Port"]`
- `graph_node_features`: `["Src IP", "Dst IP"]` (IPs excluded from tabular $X$, used strictly for graph node topological IDs)

In `configs/feature_selection/nf_cse_cic_ids2018_v2_features.json` & `nf_unsw_nb15_v2_features.json`:
- `selected_features`: Includes `"L4_DST_PORT"` (and `"L4_SRC_PORT"` in UNSW)
- `relational_features`: Includes `"L4_DST_PORT"`

### B. Numerical & Feature Pipeline Flow
1. Raw CSV Ingestion: `Dst Port` is loaded as a numerical integer column.
2. Preprocessing: Converted to float32, standardized via `StandardScaler` fitted on the Training split.
3. Model Input: Passed into `MultimodalFusionClassifier` as part of both `x_seq` (temporal vector) and `batch_graphs` / `rel_node_x` (relational vector).

---

## 3. Potential Shortcut Leakage Analysis

### A. Is `Dst Port` a Valid Feature or a Shortcut Leak?

| Dimension | Legitimacy / Argument | Shortcut Risk / Limitation |
| :--- | :--- | :--- |
| **Domain Legitimacy** | Destination ports define the target application service (Port 80/443: Web HTTP/S, Port 21: FTP, Port 22: SSH, Port 53: DNS). In real-world IDS, target port is a fundamental network flow attribute. | Specific attacks target well-known ports in static benchmark datasets (e.g. `FTP-BruteForce` exclusively on Port 21; `SSH-Bruteforce` on Port 22). Models can learn simple decision tree rules `Dst Port == 21 -> FTP-BruteForce`. |
| **Generalization Risk** | In static train/test splits, port numbers remain constant. | In evolving network traffic, attackers alter target ports (e.g., SSH running on port 2222, web server on port 8443, or port scanning across random high ports). A model heavily reliant on port memorization degrades when deployed on non-standard ports. |

### B. Decision & Recommendation
- **Decision for Phase 1**: **Preserve `Dst Port`** in the baseline feature vector to maintain consistent comparison with Phase 0 selection specs and established IDS literature.
- **Sensitivity / Ablation Recommendation**: Include an optional feature sensitivity ablation in Phase 2 comparing model performance **WITH vs WITHOUT `Dst Port`** to quantify how much predictive accuracy relies on port memorization versus payload/flow dynamics.

---

## 4. Graph Node Identifier & Transductive Leakage Validation

### A. Graph Construction Logic Inspection
Inspected `FixedRelationalEncoder` in `src/models/encoders.py` and PyTorch DataLoader logic in `src/data/loader.py`:

```python
# In src/data/loader.py:
def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
    t_X_temp = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    t_X_rel = torch.tensor(X, dtype=torch.float32)  # 2D Feature Tensor (Batch, Features)
    t_y = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(t_X_temp, t_X_rel, t_y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
```

```python
# In src/models/encoders.py (FixedRelationalEncoder):
if isinstance(node_feats, torch.Tensor):
    h_node = F.relu(self.node_proj(node_feats))
    msg = self.message_mlp(h_node)
    out = self.layer_norm(F.relu(self.global_pool(msg)))
    return out
```

### B. Findings on Transductive Graph Leakage

1. **No Global Graph Object**: In Phase 1 controlled experiments, input samples are fed into `FixedRelationalEncoder` as 2D feature tensors `(batch_size, input_dim)`.
2. **Per-Sample Node Representation**: Each flow sample's relational vector is derived independently through `node_proj` and `message_mlp` without building a global adjacency matrix across the entire dataset.
3. **No Cross-Split Node Overlap**: Because graph construction does not build a unified topological graph across train/val/test splits, test set nodes and edges **do not influence training representations**.
4. **Conclusion**: Transductive graph leakage across splits is **ABSENT** in the current Phase 1 implementation.

### C. Future Graph Phase Requirements (Phase 2+)
When explicit PyTorch Geometric graph snapshots (`Data(x, edge_index, edge_attr)`) are constructed in future research phases using `Src IP` and `Dst IP` as node endpoints:
- Node index maps must be created **independently per split or per temporal graph snapshot**.
- Message passing must be strictly bounded within the training graph snapshot to prevent test-node label propagation.
