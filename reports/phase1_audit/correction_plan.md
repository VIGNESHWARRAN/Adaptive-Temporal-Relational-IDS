# 🔧 PHASE 1 CORRECTION PLAN & VERIFICATION PROTOCOL

**Project**: Adaptive Temporal-Relational Intrusion Detection  
**Phase**: Post-Audit Correction Strategy (Execution Pending User Approval)  
**Date**: September 17, 2026  

---

## ⚠️ MANDATORY EXECUTION RULES FOR CORRECTION PHASE

1. **Wait for Approval**: No code modification or experiment re-running will occur until the user explicitly approves this audit report and correction plan.
2. **Preserve Original Artifacts**: All original results in `results/phase1_loss_experiments/` will remain completely untouched.
3. **Save Corrected Results Separately**: All corrected runs will be output to:
   ```text
   results/phase1_corrected_v1/
   ```
4. **Experimental Isolation**: Architecture (GRU-64, Relational-64, Concat Fusion, Linear-32 head), optimizer, learning rate (0.001), batch size (64), patience (5), and random seed (42) will remain **strictly fixed**. Only verified data-loader corrections and 15-class preservation logic will be applied.

---

## 1. Verified Issues & Targeted Corrections

### Correction 1: Complete 15-Class Stratified Preservation for `CSE-CIC-IDS2018`
- **Issue**: Preliminary Phase 1 sampled uniform row slices across daily files, omitting 4 rare attack classes (`SQL Injection`: 87 rows, `Brute Force -XSS`: 230 rows, `Brute Force -Web`: 611 rows, `DDOS attack-LOIC-UDP`: 1,730 rows) and reducing class count from 15 to 11.
- **Correction Logic**: Implement a stratified chunked index scanner across all 10 raw daily CSV files in `src/data/loader.py`:
  - Preserve **100% of all rare minority attack samples** ($N_c < 10,000$).
  - Stratified sample from majority classes (`Benign`, `DDOS attack-HOIC`, `DDoS attacks-LOIC-HTTP`, `DoS attacks-Hulk`) to assemble a balanced, multi-class sample representing **all 15 classes**.
  - Pass `stratify=y` to `train_test_split` to ensure every split (Train 70%, Val 15%, Test 15%) contains all 15 classes.

### Correction 2: Numerical Guard for Float Pre-clipping
- **Issue**: Converting raw DataFrame values to `float32` produced runtime overflow warnings when unclipped values exceeded $3.4 \times 10^{38}$.
- **Correction Logic**: Pre-clip raw numeric columns to $[-10^{12}, 10^{12}]$ prior to `astype(np.float32)` and apply `np.clip(-100.0, 100.0)` post-scaling.

### Correction 3: Full 9-Experiment Execution Suite
- **Affected Experiments to Rerun**:
  - `cse_cic_ids2018` $\times$ (`standard_ce`, `weighted_ce`, `focal_loss`) — **Full 15-class evaluation suite**.
  - `nf_cse_cic_ids2018_v2` $\times$ (`standard_ce`, `weighted_ce`, `focal_loss`) — Corrected output validation run.
  - `nf_unsw_nb15_v2` $\times$ (`standard_ce`, `weighted_ce`, `focal_loss`) — Corrected output validation run.

---

## 2. Target Directory Structure Post-Correction

```text
results/
├── phase1_loss_experiments/           <-- PRESERVED ORIGINAL RESULTS
│   ├── cse_cic_ids2018/
│   ├── nf_cse_cic_ids2018_v2/
│   └── nf_unsw_nb15_v2/
└── phase1_corrected_v1/              <-- NEW ISOLATED CORRECTED RESULTS
    ├── cse_cic_ids2018/
    │   ├── standard_ce/
    │   ├── weighted_ce/
    │   └── focal_loss/
    ├── nf_cse_cic_ids2018_v2/
    │   ├── standard_ce/
    │   ├── weighted_ce/
    │   └── focal_loss/
    ├── nf_unsw_nb15_v2/
    │   ├── standard_ce/
    │   ├── weighted_ce/
    │   └── focal_loss/
    └── phase1_summary/
        ├── master_results_table.csv
        └── PHASE_1_CORRECTED_EXPERIMENTAL_REPORT.md
```

---

## 3. Step-by-Step Execution Protocol (Post-Approval Only)

1. **Step 1**: Update `src/data/loader.py` with 15-class stratified sampling for `cse_cic_ids2018`.
2. **Step 2**: Execute sanity check script on loader output to confirm all 15 classes are present in Train (70%), Val (15%), and Test (15%) splits.
3. **Step 3**: Execute Phase 1 Corrected Suite outputting to `results/phase1_corrected_v1/`.
4. **Step 4**: Generate aggregate comparison report comparing original vs corrected runs.
5. **Step 5**: Present final results and walkthrough to user.
