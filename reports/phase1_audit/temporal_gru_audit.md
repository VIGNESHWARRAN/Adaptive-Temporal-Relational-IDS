# ⏱️ TEMPORAL GRU MODELING AUDIT REPORT

**Project**: Adaptive Temporal-Relational Intrusion Detection for Evolving Network Traffic  
**Phase**: Phase 1 Pre-Correction Validation & Architectural Assessment  
**Date**: September 17, 2026  

---

## 1. Executive Summary

This report audits the implementation of `FixedTemporalEncoder` in `src/models/encoders.py` and inspects the exact tensor dimensions passed to the GRU during training and evaluation.

The audit resolves whether the GRU operates as a multi-step sequential model across time or as a single-step feature transformation encoder.

---

## 2. Input Shape Inspection & Architectural Evidence

### A. Tensor Dimension Tracing
Inspected `src/data/loader.py` (Line 223) and `src/models/encoders.py` (Lines 45–61):

```python
# In src/data/loader.py:
t_X_temp = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
# Shape of t_X_temp: (batch_size, 1, input_dim)  <-- seq_len = 1
```

```python
# In src/models/encoders.py (FixedTemporalEncoder):
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)

    _, h_n = self.gru(x)     # h_n shape: (2, batch_size, 64)
    h_last = h_n[-1]         # (batch_size, 64)
    out = self.layer_norm(F.relu(self.fc_proj(h_last)))
    return out
```

### B. Empirical Finding
- The input tensor passed to the 2-layer GRU has shape **`(batch_size, 1, feature_dim)`**, where `seq_len = 1`.
- Because `seq_len = 1`, the GRU executes a single recurrent update step:

$$h_1 = \text{GRU}(x_1, h_0)$$

- The GRU does **NOT** process a sequence of multiple network flows or packets over consecutive timesteps $t_1, t_2, \dots, t_T$.

---

## 3. Terminology & Research Alignment

### A. Scientific Clarification
Because the GRU receives only a single timestep (`seq_len = 1`), it functions as a **learned non-linear feature projection module** with recurrent gated parameters, rather than modeling multi-event temporal sequences across time.

Describing this component as a *"multi-step temporal sequence encoder"* in scientific publications would be misleading.

### B. Updated Research Terminology
In project reports and research manuscripts, this component should be accurately described as:

> *"A GRU-based non-linear feature embedding encoder that processes flow-level feature representations prior to multimodal fusion."*

---

## 4. Architectural Architectural Options for Future Phases

To maintain strict experimental control in Phase 1, **the model architecture must remain unchanged**. However, we present two clear design options for the project roadmap:

### Option A: Preserve Current Architecture (Recommended for Phase 1)
- **Implementation**: Maintain `FixedTemporalEncoder` with `seq_len = 1`.
- **Rationale**: Isolates the loss function comparison without introducing sequence windowing hyperparameters (window size, step stride, temporal padding).
- **Documentation**: Update research documentation to state clearly that `FixedTemporalEncoder` performs GRU-based non-linear feature embedding on single flow records.

### Option B: Introduce Genuine Multi-Step Temporal Sequences (Proposed for Future Phases)
- **Implementation**: Group network flows by 5-tuple key `(Src IP, Dst IP, Src Port, Dst Port, Protocol)` or IP host, sort chronologically by timestamp, and construct sliding sequence windows of length $T$ (e.g., $T=10$ consecutive flows):
  $$\text{Input Shape: } (B, T, D) = (\text{batch\_size}, 10, \text{feature\_dim})$$
- **Requirements for Option B**:
  1. Chronological sorting by `Timestamp`.
  2. Sliding window construction ($T=10$, stride $S=1$).
  3. Sequence-level train/validation/test split to prevent temporal window boundary crossing.
  4. Sequence padding/truncation handling.
- **Status**: **NOT to be implemented in Phase 1** to avoid invalidating baseline loss-function comparisons. Reserved as a candidate for Phase 2/3 continual and temporal progression experiments.

---

## 5. Audit Conclusions

1. **Input Shape Confirmed**: GRU input is `(batch_size, 1, input_dim)` with `seq_len = 1`.
2. **Terminology Updated**: Component is documented as a GRU feature embedding projection.
3. **Control Maintained**: Architecture remains 100% fixed across all 9 Phase 1 experiments.
