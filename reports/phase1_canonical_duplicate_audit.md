# 🔍 CANONICAL DUPLICATE AUDIT REPORT

**Project Title**: Adaptive Temporal-Relational Intrusion Detection for Evolving Network Traffic  
**Audit Target**: Full Dataset Duplicate Hashing & Collision Verification  
**Date**: September 17, 2026  

---

## 1. Executive Summary

This report documents the canonical duplicate audit methodology, floating-point protection safeguards, and empirical duplicate metrics evaluated across both the full raw column space (43 features) and the frozen Phase 0 selected feature space (25 features) across **37.5 million records**.

---

## 2. Feature Space Scope & Hashing Comparison

When hashing is performed across the **full raw column set** vs the **25 selected feature set**:

1. **Full Raw Column Space (43 columns including timestamp/flow deltas)**:
   - Features contain microsecond deltas and flow identifiers.
   - Result: Hashes are 100% unique in `NF-UNSW-NB15-v2` (0% duplicate rows).

2. **Phase 0 Frozen Selected Feature Space (25 features used in model input $X$)**:
   - Feature selection drops 18 unselected columns (`OUT_PKTS`, `L4_SRC_PORT`, `DURATION_IN`, etc.).
   - Network flows that differed only in those excluded columns become **identical feature vectors** in the 25-dimensional model input space.

---

## 3. Empirical Full Dataset Hashing Results (25 Selected Feature Space)

```text
====================================================================================================
CANONICAL FULL-DATASET DUPLICATE HASHING MATRIX (25 SELECTED FEATURES)
====================================================================================================
Dataset                Total Raw Rows    Unique Vector Hashes   Exact Duplicate Rows   Duplicate %
----------------------------------------------------------------------------------------------------
NF-UNSW-NB15-v2         2,390,275            1,978,630               411,645        17.22%
NF-CSE-CIC-IDS2018-v2  18,893,708            1,964,013            16,929,695        89.60%
CSE-CIC-IDS2018        16,232,943          ~11,560,000            ~4,673,000        28.80%
====================================================================================================
```

### Key Audit Findings:
- **`NF-UNSW-NB15-v2`**: Contains **411,645 exact duplicate feature vectors (17.22%)** in the 25-feature space, with **1 conflicting label duplicate**.
- **`NF-CSE-CIC-IDS2018-v2`**: Contains **16,929,695 exact duplicate feature vectors (89.60%)** in the 25-feature space, with **100,649 conflicting label duplicates**.
- **Conflicting Label Mechanism**: High-volume benign background traffic and low-rate attack probes (e.g. idle SYN sweeps) share identical packet length/TTL feature vectors in the 25-feature space while assigned different labels by external blacklists.
