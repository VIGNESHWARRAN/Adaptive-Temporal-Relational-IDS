# 🛡️ PHASE 1 SPLIT LEAKAGE AUDIT REPORT

**Project Title**: Adaptive Temporal-Relational Intrusion Detection for Evolving Network Traffic  
**Audit Target**: Recreated Phase 1 Train/Val/Test Split Data Leakage Verification  
**Date**: September 17, 2026  

---

## 1. Recreated Phase 1 Split Parameters

The exact Phase 1 split was recreated in [rigorous_phase1_audit_validator.py](file:///C:/Users/Sai%20Raman/.gemini/antigravity/brain/4f4a112d-c31e-4184-b113-0ffb8926b382/scratch/rigorous_phase1_audit_validator.py) using:
- **`sample_limit`**: 100,000 samples per dataset
- **`random_state`**: 42 (Fixed NumPy and PyTorch seed)
- **Split Ratios**: 70% Train, 15% Validation, 15% Test
- **Stratification**: `stratify=y` on label array
- **Features**: Phase 0 frozen selected features (25 features for NF datasets, 28 for CSE-CIC)

---

## 2. Empirical Split Leakage Vector Audit Results (100,000 Sample Limit)

| Dataset | Train vs Val Hash Overlap | Train vs Test Hash Overlap | Val vs Test Hash Overlap | Preprocessor Scope | Assessment / Status |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **NF-UNSW-NB15-v2** | **219** | **245** | **62** | Fitted strictly on Train | **DUPLICATE LEAKAGE** |
| **NF-CSE-CIC-IDS2018-v2** | **4,100** | **4,120** | **2,339** | Fitted strictly on Train | **HIGH DUPLICATE LEAKAGE** |
| **CSE-CIC-IDS2018** | **611** | **599** | **298** | Fitted strictly on Train | **DUPLICATE LEAKAGE** |

---

## 3. Empirical 5-Tuple Flow Key Overlap Metrics (`NF-UNSW-NB15-v2`)

- **Total Unique 5-Tuple Flow Keys in 100k**: 80,490
- **Train Flow Keys**: 56,996 | **Val Flow Keys**: 13,414 | **Test Flow Keys**: 13,444
- **Train vs Val Flow Overlap Keys**: **1,674** (12.48% of Val keys)
- **Train vs Test Flow Overlap Keys**: **1,677** (12.47% of Test keys)
- **Val vs Test Flow Overlap Keys**: **1,326** (9.86% of Test keys)
- **Test Rows with Train Flow Key**: **3,227 / 15,000 Test rows (21.51% of Test rows)**

---

## 4. Scientifically Defensible Leakage Statement

> *"Recreating the exact 70/15/15 split reveals that random row-wise splitting introduces exact feature vector duplicate overlap across partitions (4,120 duplicate vectors between Train and Test in NF-CSE-CIC-IDS2018-v2, 245 in NF-UNSW-NB15-v2). Additionally, 12.47% of 5-tuple flow keys overlap between Train and Test in NF-UNSW-NB15-v2, affecting 21.51% of Test set rows."*
