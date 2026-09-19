# 🛡️ Multimodal Adaptive Temporal-Relational Classifier for Network Intrusion Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![Platform: Colab / Local](https://img.shields.io/badge/platform-Google%20Colab%20%7C%20Local-orange.svg)]()
[![License: Academic Research](https://img.shields.io/badge/license-Academic%20Research-green.svg)]()

> **Project:** *Multimodal Adaptive Temporal-Relational Classifier for Network Intrusion Detection: Investigating Classical, Quantum, Gradient-Adaptive, and Continual Learning Approaches*

---

## 📖 Overview

Conventional Network Intrusion Detection Systems (NIDS) treat network flows as isolated tabular records, ignoring crucial **temporal dynamics** (inter-arrival times, sequence ordering, burst behavior) and **relational interactions** (communication topology $G_t = (V_t, E_t)$, entity connectivity).

This repository provides an enterprise-grade research framework incorporating:
1. **Multi-Source Dataset Ingestion & Preprocessing Engine:** Resilient, chunked ingestion supporting NetFlow v9 (`NF-CSE-CIC-IDS2018-v2`, `NF-UNSW-NB15-v2`) and deep bidirectional flow statistics (`CSE-CIC-IDS2018`).
2. **Forensic Audit & Data Leakage Verification Protocol:** Multi-stage audit suites resolving cross-file flow leakage, exact duplicate row reconciliation, fallback-string collapse, and rare-class stratification guarantees.
3. **Gradient-Based Adaptive Hybrid Loss ($L_{\text{hybrid}}$):** A dynamic loss function balancing Cross-Entropy, Focal Loss ($\gamma=2.0$), and Class-Weighted Cross-Entropy via autograd reference parameter gradient norm balancing, weight clipping, and EMA smoothing.
4. **Multimodal Fusion Architecture Suite (18 Experiments):** 6 controlled fusion mechanisms (`Concatenation`, `Gated`, `Bilinear`, `Cross-Attention`, `Decision Ensemble`, `Quantum PQC`) evaluated across 3 benchmark datasets.
5. **Phase 1 Baseline Controlled Experimentation Harness:** 9 baseline loss experiments comparing unweighted, weighted, and focal losses under fixed architectures.

---

## 🔬 Forensic Audits & Dataset Integrity Verification

Before model training, a multi-stage forensic audit cycle was conducted to ensure strict empirical data integrity, leak-free splits, and reproducible metrics.

### 1. Cross-File Flow Leakage & Fallback Key Audit
- **Finding:** Initial key matching indicated $99.90\%$ cross-file key overlap in `CSE-CIC-IDS2018`. Forensic inspection revealed that $99.90\%$ of overlap was driven by **artificial fallback-string collapse** (due to missing 5-tuple flow identifiers across 9 of 10 raw CSV files), rather than genuine host-to-host flow leakage.
- **Audit Decision:** Marked validation status as:  
  `VALIDATION INCOMPLETE — GENUINE CROSS-FILE FLOW LEAKAGE NOT ESTABLISHED`
- **Resolution:** Fallback keys were separated completely from genuine 5-tuple flow keys (`02-20-2018.csv`). Temporal sequence ordering and flow key integrity are strictly preserved.

### 2. Exact Duplicate & Conflicting-Label Reconciliation
Across the total population of **16,232,943** records in `CSE-CIC-IDS2018`:
- **Total Rows:** $16,232,943$
- **Unique 28-Feature Vectors:** $11,464,646$
- **Exact Duplicate Rows:** $4,768,297$
- **Duplicate Feature Groups:** $688,600$
- **Conflicting-Label Feature Groups:** $68,182$
- **Definition A (Disambiguated Conflicting Rows):** $177,674$ rows (rows with conflicting label assignments within duplicate groups).
- **Definition B (Conflicting Group Total Population):** $1,570,760$ rows (total count of all rows belonging to conflicting feature groups).

### 3. Rare-Class Stratification & Sampling Quota Guarantees
- **Total Class Count:** **15 Total Classes** ($1 \text{ Benign} + 14 \text{ Attack Classes}$).
- **Attack Classes (14):** `Bot`, `DDoS attacks-LOIC-HTTP`, `DoS attacks-DDOS`, `DoS attacks-GoldenEye`, `DoS attacks-Hulk`, `DoS attacks-SlowHTTPTest`, `DoS attacks-Slowloris`, `FTP-BruteForce`, `Infiltration`, `SSH-Bruteforce`, `Brute Force -Web`, `Brute Force -XSS`, `SQL Injection`, `DDoS attack-HOIC`.
- **Sampling Quota Standard:** $100,000$ samples targeted per dataset prior to retaining rare classes, ensuring all 15 classes exist across Train ($70\%$), Validation ($15\%$), and Test ($15\%$) splits without rare-class dropouts.

---

## ⚡ Gradient-Based Adaptive Hybrid Loss Function

The **Gradient-Based Adaptive Hybrid Loss** dynamically adjusts loss component weights during training based on the relative magnitude of gradients backpropagated through a common trainable reference parameter set $\Theta_{\text{ref}}$.

### 1. Mathematical Formulation
$$L_{\text{hybrid}} = \alpha \cdot L_{\text{CE}} + \beta \cdot L_{\text{Focal}} + \gamma \cdot L_{\text{WeightedCE}}$$

Where:
- $L_{\text{CE}}$: Standard Unweighted Cross-Entropy Loss
- $L_{\text{Focal}}$: Multiclass Focal Loss ($\gamma=2.0$)
- $L_{\text{WeightedCE}}$: Inverse Class Frequency Weighted Cross-Entropy Loss

### 2. Reference Parameter Gradient Norm Calculation
For each loss component $i \in \{\text{CE}, \text{Focal}, \text{WCE}\}$, the L2 gradient norm over reference parameters $\Theta_{\text{ref}}$ is computed via autograd:
$$g_i = \left\| \nabla_{\Theta_{\text{ref}}} L_i \right\|_2 = \sqrt{ \sum_{\theta \in \Theta_{\text{ref}}} \left( \frac{\partial L_i}{\partial \theta} \right)^2 + \epsilon }$$

- **Reference Parameter Selection ($\Theta_{\text{ref}}$):** Extracted via `get_reference_parameters()` in `MultimodalFusionClassifier`:
  - For Non-Ensemble models (`concat`, `gated`, `bilinear`, `cross_attention`, `quantum_pqc`): $\Theta_{\text{ref}} = \text{classifier\_head.parameters()}$
  - For Ensemble models (`decision_ensemble`): $\Theta_{\text{ref}} = \text{fusion\_module.parameters()}$

### 3. Gradient Ratio Inversion & Prior Weighting
$$g_{\text{mean}} = \frac{g_{\text{CE}} + g_{\text{Focal}} + g_{\text{WCE}}}{3}$$

$$r_i = \left( \frac{g_{\text{mean}}}{g_i + \epsilon} \right)^\eta, \quad \eta = 0.5, \; \epsilon = 1e-8$$

$$u_i = p_i \cdot r_i, \quad (p_{\text{CE}}=0.50, \; p_{\text{Focal}}=0.35, \; p_{\text{WCE}}=0.15)$$

$$w_i^{\text{raw}} = \frac{u_i}{\sum_j u_j}$$

### 4. Weight Safeguards (Clipping & EMA Smoothing)
1. **Weight Clipping:** Weights are clamped to $[\text{min\_weight}=0.05, \, \text{max\_weight}=0.80]$ and re-normalized so $\sum w_i = 1.0$.
2. **Exponential Moving Average (EMA):**
   $$w_t^{\text{smoothed}} = 0.90 \cdot w_{t-1}^{\text{smoothed}} + 0.10 \cdot w_t^{\text{clipped}}$$
3. **Autograd Graph Isolation:** Weights $\alpha, \beta, \gamma$ are detached before computing $L_{\text{hybrid}}$, preventing higher-order computational graph overhead.

---

## 🎛️ Multimodal Fusion Architecture Suite (18 Experiments)

Evaluates 6 multimodal fusion mechanisms combining GRU temporal sequence representations and graph relational representations across 3 benchmark datasets:

$$\mathbf{3 \text{ Datasets}} \times \mathbf{6 \text{ Fusion Methods}} = \mathbf{18 \text{ Controlled Experiments}}$$

| Fusion Method | Identifier | Mathematical Mechanism | Reference Parameter Set ($\Theta_{\text{ref}}$) |
| :--- | :--- | :--- | :--- |
| **Concatenation** | `concat` | $\mathbf{f} = [\mathbf{h}_{\text{temp}} \,\|\, \mathbf{h}_{\text{rel}}]$ | `classifier_head` (64 $\to$ 32 $\to$ $C$) |
| **Gated Fusion** | `gated` | $\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{h}_{\text{temp}} \,\|\, \mathbf{h}_{\text{rel}}])$, $\mathbf{f} = \mathbf{g} \odot \mathbf{h}_{\text{temp}} + (\mathbf{1}-\mathbf{g}) \odot \mathbf{h}_{\text{rel}}$ | `classifier_head` |
| **Bilinear Fusion** | `bilinear` | $\mathbf{f} = \mathbf{h}_{\text{temp}} \mathbf{W}_b \mathbf{h}_{\text{rel}}^T$ | `classifier_head` |
| **Cross-Attention** | `cross_attention` | $\mathbf{Q}=\mathbf{h}_{\text{temp}}\mathbf{W}_q, \mathbf{K}=\mathbf{h}_{\text{rel}}\mathbf{W}_k, \mathbf{V}=\mathbf{h}_{\text{rel}}\mathbf{W}_v$ | `classifier_head` |
| **Decision Ensemble**| `decision_ensemble` | $\mathbf{p} = \text{Softmax}(\mathbf{p}_{\text{temp}}) + \text{Softmax}(\mathbf{p}_{\text{rel}})$ | `fusion_module` |
| **Quantum PQC** | `quantum_pqc` | Parameterized Quantum Circuit feature rotation / inner product | `classifier_head` |

---

## 🗂️ Repository Architecture

```text
adaptive-temporal-relational-ids/
├── README.md                           # Documentation, audit findings & execution guide
├── requirements.txt                    # Python dependencies
├── configs/
│   ├── datasets_config.yaml            # Dataset registry (URLs, mirrors, Kaggle slugs)
│   └── colab_config.yaml               # Google Colab & Google Drive paths
├── src/
│   ├── data/
│   │   └── loader.py                   # Data split (70/15/15), scaler fit on train ONLY, class weights
│   ├── models/
│   │   ├── encoders.py                 # FixedTemporalEncoder (GRU) & FixedRelationalEncoder
│   │   ├── fusion.py                   # 6 Multimodal Fusion modules (Concat, Gated, Bilinear, Attention, Ensemble, Quantum)
│   │   ├── classifiers.py              # MultimodalFusionClassifier & get_reference_parameters()
│   │   └── losses.py                   # Standard CE, Class-Weighted CE, Focal Loss, GradientAdaptiveHybridLoss
│   ├── evaluation/
│   │   ├── leakage_checker.py          # Mandatory 7-point data leakage protocol
│   │   └── trainer.py                  # ExperimentTrainer with adaptive loss diagnostic logging & plot exports
│   └── phase0/                         # Forensic audit, duplicate reconciliation & leakage verification tools
├── scripts/
│   ├── smoke_test_hybrid_fusion.py     # Smoke test verifying all 6 fusion methods with adaptive hybrid loss
│   ├── run_hybrid_fusion_experiments.py# Master harness for 18 Gradient-Adaptive Hybrid Loss experiments
│   ├── run_phase1_experiments.py       # Harness for 9 Phase 1 baseline loss experiments
│   └── generate_phase1_report.py       # Aggregates Phase 1 baseline results
├── experiments/
│   └── gradient_adaptive_hybrid/       # Output directory for 18 fusion experiments & master reports
│       ├── master_summary_report.csv   # Master metrics CSV across all 18 experiments
│       └── master_summary_report.md    # Master Markdown summary table
└── results/
    └── phase1_corrected_v1/            # Preserved Phase 1 baseline experiment results
```

---

## 🚀 Quickstart & Execution Guide

### 1. Verification Smoke Test
Verify that all 6 fusion architectures and `GradientAdaptiveHybridLoss` operate cleanly without device or shape mismatch errors:

```bash
python scripts/smoke_test_hybrid_fusion.py
```

*Output:* `[ALL SMOKE TESTS PASSED] GradientAdaptiveHybridLoss works with all 6 fusion methods!`

### 2. Execute the 18 Multimodal Fusion Experiments
Run the full suite across 3 datasets and 6 fusion methods with Gradient-Based Adaptive Hybrid Loss:

```bash
python scripts/run_hybrid_fusion_experiments.py --dataset all --fusion all --epochs 15 --sample-limit 100000
```

### 3. Run Specific Single Experiment
```bash
# Run specific dataset and fusion method:
python scripts/run_hybrid_fusion_experiments.py --dataset nf_cse_cic_ids2018_v2 --fusion concat

# Run all 6 fusion methods for UNSW-NB15:
python scripts/run_hybrid_fusion_experiments.py --dataset nf_unsw_nb15_v2 --fusion all
```

---

## 📊 Evaluation Outputs & Master Reports

Each experiment in `experiments/gradient_adaptive_hybrid/{dataset}/{fusion}/` produces:
- `metrics.json`: Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Training Time, Best Epoch, and Adaptive Loss Summaries.
- `adaptive_loss_diagnostics.csv`: Step-by-step logs of $L_{\text{CE}}, L_{\text{Focal}}, L_{\text{WCE}}, L_{\text{hybrid}}$, gradient norms $g_i$, adaptive weights $(\alpha, \beta, \gamma)$, clipping events, and fallbacks.
- `{dataset}_{fusion}_cm.png`: Publication-ready Confusion Matrix heatmap.
- `{dataset}_{fusion}_loss_curve.png`: Training & Validation loss curves with early stopping marker.
- `best_model.pt`: Model weights checkpoint at best validation loss.
- `master_summary_report.csv` & `master_summary_report.md`: Combined comparison table summarizing all 18 experiments.
