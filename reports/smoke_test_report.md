# 🧪 Phase 1 Loader & Training Smoke Test Report

**Date**: September 19, 2026  
**Status**: **`PASSED SUCCESSFULLY`**

## Execution Summary
- **Dataset**: `CSE-CIC-IDS2018`
- **Loss Function**: `standard_ce` (Standard Cross-Entropy)
- **Epochs Tested**: `1 Epoch`
- **Output Directory**: `results/phase1_corrected_v1/cse_cic_ids2018/standard_ce`

## Empirical Results
- **Extracted Total Rows**: `141,717 rows`
- **Classes Represented**: `15 / 15 Classes`
- **Train / Val / Test Counts**: `99,201 / 21,258 / 21,258`
- **Split Manifest Generated**: `results/phase1_corrected_v1/cse_cic_ids2018/standard_ce/split_manifest.csv` (141,717 rows)
- **Test Loss**: `0.1750`
- **Test Accuracy**: `0.9424` (94.24%)
- **Macro F1 Score**: `0.7818`

## Verification Confirmations
1. **Zero NaN / Inf Loss Crashes**: Finite loss computed across all batches.
2. **Label Encoding Identity**: 15 distinct classes mapped cleanly without collision.
3. **Scaler & Imputation Leakage Protection**: Fitted strictly on Train set (`scaler_fitted_on_train_only = True`).
4. **Manifest Export**: Source file and row index preserved for all 141,717 records.
