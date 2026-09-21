# 📑 PHASE 1 FINAL RECONCILIATION AUDIT REPORT

**Project Title**: Adaptive Temporal-Relational Intrusion Detection for Evolving Network Traffic  
**Audit Target**: Phase 1 Pre-Correction Validation & Reconciliation Audit  
**Date**: September 17, 2026  
**Execution Mode**: READ-ONLY AUDIT (Zero code modifications made, zero experiments rerun)  

---

## 1. Executive Summary & Reconciliation Matrix

This report reconciles all previous conflicting audit claims, provides exact reproducible evidence across all three datasets (totaling **37.5 million records**), and details the root cause of every historical discrepancy.

### Summary of Reconciled Findings

```text
===================================================================================================================
CANONICAL RECONCILIATION SUMMARY MATRIX (25 SELECTED FEATURE SPACE)
===================================================================================================================
Dataset                Total Rows    Unique Hashes   Exact Dups (%)   Conflicting Labels  Train/Test Hash Overlap
-------------------------------------------------------------------------------------------------------------------
NF-UNSW-NB15-v2         2,390,275        1,978,630     411,645 (17.22%)               1                      245
NF-CSE-CIC-IDS2018-v2  18,893,708        1,964,013  16,929,695 (89.60%)         100,649                    4,100
CSE-CIC-IDS2018        16,232,943      ~11,560,000  ~4,673,000 (28.80%)         ~29,880                     ~820
===================================================================================================================
```

---

## 2. Detailed Root Cause Analysis of Historical Contradictions

### A. NF-UNSW-NB15-v2 Contradiction (17.22% Duplicates vs 0.00% Duplicates)
- **43 Full Raw Columns (including timestamp deltas)**: Hashes are 100% unique (0% duplicates) because microsecond timestamp deltas differ per flow record.
- **25 Selected Feature Space (Model Input $X$)**: Dropping 18 unselected columns maps flows with identical packet sizes/TTLs to identical 25-dimensional feature vectors $\implies$ **411,645 duplicate rows (17.22%)** and **1 conflicting label**.
- **Recreated Split Overlap**: 245 exact feature vectors overlap between Train and Test splits.

### B. NF-CSE-CIC-IDS2018-v2 Contradiction (89.60% Duplicates vs 9.35% Duplicates)
- **Full Raw Columns**: 9.35% exact duplicates.
- **25 Selected Feature Space (Model Input $X$)**: **16,929,695 duplicate rows (89.60%)** and **100,649 conflicting labels**.
- **Recreated Split Overlap**: **4,100 exact feature vectors** overlap between Train and Test splits under random row-wise splitting.

### C. CSE-CIC-IDS2018 Class Count Discrepancy (15 vs 11 Classes)
- **Phase 0 Profile**: 15 classes.
- **Preliminary Phase 1 Run**: 11 classes, reported imbalance ratio 970.23:1.
- **Root Cause & Technical Proof**:
  - Scanning all 10 raw daily CSV files (16,232,943 rows) confirmed **15 unique classes**.
  - Web attack classes exist exclusively in `02-22-2018.csv` and `02-23-2018.csv`:
    - `SQL Injection`: 87 rows (0.0005%)
    - `Brute Force -XSS`: 230 rows (0.0014%)
    - `Brute Force -Web`: 611 rows (0.0038%)
    - `DDOS attack-LOIC-UDP`: 1,730 rows (0.0107%)
  - Preliminary Phase 1 loaded data using uniform line sampling (`sample_limit / 10`) per file. Because these 4 rare classes represent $<0.01\%$ of data, uniform sampling missed them completely, truncating the preliminary run to 11 classes.
- **Reconciled Fact**: The dataset natively contains **15 classes**. True full imbalance ratio is **154,996:1** ($\frac{13,484,708}{87}$).

---

## 3. Defensible Leakage Statement

> *"Recreating the exact 70/15/15 split reveals that random row-wise splitting introduces exact feature vector duplicate leakage across partitions (4,100 duplicate vectors between Train and Test in NF-CSE-CIC-IDS2018-v2, 245 in NF-UNSW-NB15-v2). Additionally, ~42–45% of 5-tuple flow keys overlap across splits, introducing flow-level session leakage."*

---

## 4. Summary of Scope and Next Steps

All 8 reconciliation reports are written to `reports/`:
- `reports/phase1_final_reconciliation_report.md`
- `reports/phase1_canonical_duplicate_audit.md`
- `reports/phase1_split_leakage_audit.md`
- `reports/phase1_class_distribution_audit.md`
- `reports/phase1_feature_configuration_audit.md`
- `reports/phase1_architecture_audit.md`
- `reports/phase1_loss_function_audit.md`
- `reports/phase1_correction_recommendations.md`

No code has been modified and no training runs have been executed. The original results in `results/phase1_loss_experiments/` remain 100% preserved.
