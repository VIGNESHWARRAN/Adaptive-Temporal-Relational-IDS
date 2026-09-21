# 🔌 PHASE 1 FEATURE CONFIGURATION AUDIT REPORT

**Project Title**: Adaptive Temporal-Relational Intrusion Detection for Evolving Network Traffic  
**Audit Target**: Feature Selection, `Dst Port` Shortcut Analysis & Input Schema Invariance  
**Date**: September 17, 2026  

---

## 1. Feature Selection Configurations

- **CSE-CIC-IDS2018**: 28 selected features (out of 84 raw columns)
- **NF-CSE-CIC-IDS2018-v2**: 25 selected features (out of 43 raw columns)
- **NF-UNSW-NB15-v2**: 25 selected features (out of 43 raw columns)

Pure identity features (`Src IP`, `Dst IP`, `Timestamp`, `Flow ID`, `Label`) are excluded from tabular feature matrix $X$ across all three datasets.

---

## 2. `Dst Port` Shortcut Leakage & Inference Availability Assessment

1. **Role of `Dst Port`**: Included in `selected_features` and `relational_features` as a numerical destination port attribute.
2. **Inference Availability**: **YES**. Destination port is a standard layer-4 header field present in every live network packet and NetFlow record.
3. **Shortcut Memorization Risk**:
   - Attacks such as `FTP-BruteForce` (Port 21) or `SSH-Bruteforce` (Port 22) have constant target ports in static benchmark datasets.
   - Decision models can memorize `Dst Port == 21 -> FTP-BruteForce`.
4. **Recommendation for Baseline & Future Experiments**:
   - **Phase 1 Decision**: **Preserve `Dst Port`** to maintain experimental consistency with Phase 0 frozen specifications.
   - **Phase 2 Ablation Proposal**: Run a feature sensitivity experiment (**WITH vs WITHOUT `Dst Port`**) to measure model reliance on port lookup vs payload/flow dynamics.
