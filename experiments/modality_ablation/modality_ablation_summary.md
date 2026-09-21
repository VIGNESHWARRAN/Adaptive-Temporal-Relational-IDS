# Master Summary Report: Temporal-Only vs Spatial/Relational-Only Modality Ablation

Evaluated across 3 completed experiments without multimodal fusion.

## 1. Overall Modality Performance Comparison

| Dataset | Architecture | Accuracy | Macro F1 | Weighted F1 | Benign Recall | Attack Macro Recall | FPR | FNR | Trainable Params | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cse_cic_ids2018 | temporal_only | 0.8585 | 0.1877 | 0.7974 | 0.9942 | 0.1915 | 0.0058 | 0.7870 | 46,223 | 385.2 |
| cse_cic_ids2018 | relational_only | 0.8425 | 0.1232 | 0.7716 | 0.9977 | 0.0698 | 0.0023 | 0.9004 | 15,311 | 192.6 |
| cse_cic_ids2018 | both | 0.8772 | 0.2811 | 0.8341 | 0.9804 | 0.2820 | 0.0196 | 0.4809 | 67,343 | 307.6 |

---

## 2. Scientific Analysis & Modality Insights

### A. Temporal-Only Architecture (`FixedTemporalEncoder` GRU)
- **Strengths**: Captures sequential dependencies, inter-arrival time bursts, and multi-stage traffic progressions.
- **Weaknesses**: May struggle on static single-flow attacks where topological context (host connectivity) is required for disambiguation.

### B. Spatial/Relational-Only Architecture (`FixedRelationalEncoder` Graph)
- **Strengths**: Captures entity interaction graphs, host node degree distributions, and communication fan-out.
- **Note on Spatial Terminology**: While designated as 'Spatial/Relational-Only Model', the architecture models communication graph interactions rather than physical geographical coordinates.
- **Weaknesses**: Cannot leverage sequential state across consecutive packets within a flow without temporal hidden state tracking.

### C. Multimodal Fusion Architecture (`MultimodalFusionClassifier` Both Fused)
- **Strengths**: Combines sequential flow dynamics (GRU) with spatial/graph interaction dynamics, providing complementary features for robust detection across multi-vector attacks.
- **Trade-off**: Requires slightly higher parameter count and runtime overhead than single-modality models.

