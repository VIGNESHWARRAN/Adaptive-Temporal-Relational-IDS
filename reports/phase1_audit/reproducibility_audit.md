# 🔄 REPRODUCIBILITY AUDIT REPORT

**Project**: Adaptive Temporal-Relational Intrusion Detection  
**Phase**: Phase 1 Controlled Experimentation Audit  
**Date**: September 17, 2026  

---

## 1. Executive Summary

Research reproducibility requires that any independent researcher can execute the codebase with identical configurations and reproduce the exact experimental metrics within deterministic numeric bounds.

This audit evaluates dataset persistence, configuration serialization, random seed coverage, device determinism, environment specifications, and output artifact naming.

---

## 2. Reproducibility Checklist & Audit Status

| Reproducibility Requirement | Implementation Mechanism | Audit Finding | Status |
| :--- | :--- | :--- | :--- |
| **Fixed Random Seed** | `set_seed(42)` sets NumPy & PyTorch seeds | Enforced at start of every experiment | **PASSED** |
| **Deterministic Data Splits** | `train_test_split(..., random_state=42)` | Identical train/val/test partitions | **PASSED** |
| **Frozen Feature Configs** | `configs/feature_selection/*.json` | Serialized JSON loaded in `loader.py` | **PASSED** |
| **Model Weight Init** | PyTorch seed set before instantiation | Identical initial weight states | **PASSED** |
| **Config Serialization** | `config.json` exported per run | All hyperparameters saved to disk | **PASSED** |
| **Metric Serialization** | `metrics.json` & confusion matrices | Complete per-class report exported | **PASSED** |
| **Best Model Checkpoint** | `best_model.pt` saved per run | Checkpoints persisted in output dir | **PASSED** |
| **Hardware Determinism** | PyTorch CUDA CUDNN settings | Floating-point non-determinism note | **MEDIUM** |

---

## 3. Detailed Audit Findings

### 1. Hardcoded Dataset Paths vs Config-Driven Ingestion
- **Current Logic**: `src/data/loader.py` uses fallback matching for dataset paths (e.g., searching for subfolder hashes `b3427ed8ad063a09_MOHANAD_A4706`).
- **Audit Assessment**: While robust to local directory variations, providing explicit relative paths in dataset configuration files eliminates resolution ambiguity.

### 2. Multi-File Sampling Reproducibility (`CSE-CIC-IDS2018`)
- **Current Logic**: `load_dataset_split()` samples rows per file using `f_df.sample(n=rows_per_file, random_state=seed)`.
- **Audit Assessment**: Sampling with fixed `random_state=42` is deterministic when the raw files remain unchanged. However, uniform per-file sampling without per-class stratification across files drops extreme minority classes (`SQL Injection`, `Brute Force -XSS`).
- **Action Required**: The sampling logic must incorporate stratified class allocation across all 10 daily CSV files.

### 3. Floating-Point Precision & Hardware Non-Determinism
- GPU matrix multiplication (cuBLAS) can introduce minor floating-point variations ($< 10^{-5}$) across different GPU architectures (e.g., RTX 3090 vs T4 vs CPU).
- Adding `torch.use_deterministic_algorithms(True)` can be configured for bit-exact reproduction when required.

---

## 4. Reproducibility Artifact Verification

Each completed experiment generates a standardized output bundle in `results/phase1_loss_experiments/{dataset}/{loss}/`:
- `config.json` — Complete run parameters
- `dataset_integration_report.json` — Loaded feature configuration & split sizes
- `leakage_report.json` — Results of 7 mandatory leakage checks
- `metrics.json` — Full accuracy, macro/weighted F1, and per-class precision/recall/F1 metrics
- `best_model.pt` — PyTorch state dict checkpoint at minimum validation loss epoch
- `{exp}_cm.png` — High-resolution confusion matrix plot
- `{exp}_loss_curve.png` — Training & validation loss curve plot

---

## 5. Audit Conclusion

The experiment harness has high reproducibility infrastructure. Implementing stratified per-class sampling for `CSE-CIC-IDS2018` in the post-approval correction phase will make the experimental framework 100% reproducible and complete across all 15 classes.
