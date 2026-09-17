# 🛡️ Adaptive Temporal-Relational Intrusion Detection System (IDS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Colab / Local](https://img.shields.io/badge/platform-Google%20Colab%20%7C%20Local-orange.svg)]()
[![License: Academic Research](https://img.shields.io/badge/license-Academic%20Research-green.svg)]()

> **Project:** *Adaptive Temporal-Relational Intrusion Detection for Evolving Network Traffic: Investigating Classical, Quantum, and Continual Learning Approaches*

---

## 📖 Overview

Conventional intrusion detection systems treat network flows as independent tabular records, discarding critical **temporal sequences** (event ordering, inter-arrival times, multi-stage burst dynamics) and **relational topology** (communication graphs $G_t = (V_t, E_t)$, entity interactions, host connectivity).

This repository provides the core data engineering, architecture modules, and experimental harness:
1. **Multi-Source Dataset Ingestion Engine:** Resilient, chunked downloading supporting direct HTTP/HTTPS, Zenodo, and Kaggle API.
2. **Standardized NetFlow & Deep Flow Schema Alignment:** Supporting Tier 1 core enterprise NetFlow (`NF-CSE-CIC-IDS2018-v2`, `NF-UNSW-NB15-v2`), deep flow statistics (`CSE-CIC-IDS2018`), and cross-domain IoT benchmarks (`NF-ToN-IoT-v2`, `CICIoT2023`).
3. **Automated Metadata Profiler & Leakage Checker:** Inspects schemas, null rates, record counts, and enforces a mandatory 7-point data leakage protocol.
4. **Phase 1 Controlled Loss-Function Experimentation Harness:** Evaluates **Standard Cross Entropy**, **Class-Weighted Cross Entropy**, and **Focal Loss** ($\gamma=2.0$) across 3 benchmark datasets under strictly frozen model architectures.
5. **Zero-Laptop-Spec Google Colab Workflow:** Ready-to-run Colab notebooks with automatic Google Drive persistence.

---

## 🗂️ Repository Architecture

```text
adaptive-temporal-relational-ids/
├── .gitignore                          # Excludes raw CSVs, PCAPs, checkpoints, virtualenvs
├── README.md                           # Documentation & execution guide
├── requirements.txt                    # Lightweight dependencies
├── configs/
│   ├── datasets_config.yaml            # Dataset registry (URLs, mirrors, Kaggle slugs, tiers)
│   └── colab_config.yaml               # Google Colab & Google Drive paths
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                   # Data split (70/15/15), scaler fit on train ONLY, class weights
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoders.py                 # FixedTemporalEncoder (GRU) & FixedRelationalEncoder
│   │   ├── fusion.py                   # ConcatenationFusion (Concat 128 -> Linear 64 -> BatchNorm -> ReLU)
│   │   ├── classifiers.py              # MultimodalFusionClassifier with fixed head (64->32->num_classes)
│   │   └── losses.py                   # Standard CE, Class-Weighted CE, FocalLoss (gamma=2)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── leakage_checker.py          # Mandatory 7-point data leakage protocol
│   │   └── trainer.py                  # PyTorch trainer, early stopping, loss curves, confusion matrices
│   ├── download/
│   │   ├── dataset_registry.py         # Registry loader & tier filter
│   │   └── downloader.py               # Resumable HTTP, KaggleHub & Zenodo downloader
│   └── metadata/
│       ├── extractor.py                # Schema, null rate, and class distribution profiler
│       ├── temporal_relational_eval.py # Relational/Temporal feature compatibility evaluator
│       └── reporter.py                 # Generates Markdown and JSON metadata reports
├── scripts/
│   ├── download_datasets.py            # CLI script to download datasets
│   ├── extract_metadata.py             # CLI script to profile datasets and create reports
│   ├── run_all_pipeline.py             # Master pipeline orchestrator
│   ├── run_phase1_experiments.py       # Controlled CLI harness for Phase 1 9-experiment suite
│   └── generate_phase1_report.py       # Aggregates Phase 1 metrics into master table & report
├── notebooks/
│   ├── 01_download_and_metadata_colab.ipynb   # Metadata & dataset ingestion notebook
│   └── 02_phase1_loss_experiments_colab.ipynb  # Interactive Phase 1 loss experiments notebook
├── results/
│   └── phase1_loss_experiments/        # Phase 1 experimental metrics, confusion matrices & reports
└── metadata_reports/                   # Output folder for generated metadata JSON & MD reports
```

---

## ⚡ Phase 1: Controlled Loss-Function Experimentation

Phase 1 conducts a controlled comparative study of 3 classification loss functions across 3 benchmark datasets:

**3 Datasets × 3 Loss Functions = 9 Controlled Experiments**

### Experimental Controls
- **Datasets:** `NF-CSE-CIC-IDS2018-v2`, `NF-UNSW-NB15-v2`, `CSE-CIC-IDS2018`
- **Loss Functions:** Standard Cross Entropy (`standard_ce`), Class-Weighted Cross Entropy (`weighted_ce`), Focal Loss (`focal_loss`, $\gamma=2.0$)
- **Fixed Model Architecture:** `FixedTemporalEncoder` (GRU-64d) + `FixedRelationalEncoder` (64d) + `ConcatenationFusion` (64d) + Classifier Head (`Linear 64->32 -> ReLU -> Dropout(0.2) -> Linear 32->num_classes`)
- **Strict Invariants:** Preprocessing, feature selection, 70/15/15 splits, random seed (42), optimizer (Adam), learning rate (0.001), batch size, and evaluation criteria remain **strictly identical** across experiments.

---

## 🚀 Quickstart: Running on Google Colab (Recommended)

1. Open **[Google Colab](https://colab.research.google.com/)**.
2. Open [`notebooks/02_phase1_loss_experiments_colab.ipynb`](notebooks/02_phase1_loss_experiments_colab.ipynb).
3. Execute the cells to run Phase 1 loss experiments remotely on GPU/CPU:
   ```bash
   # Run all 9 controlled experiments
   !python scripts/run_phase1_experiments.py --dataset all --loss all --epochs 15
   
   # Generate Master Summary Report & Master Table
   !python scripts/generate_phase1_report.py results/phase1_loss_experiments
   ```

---

## 💻 Running Locally / CLI Usage

### 1. Setup Virtual Environment

```bash
# Clone or navigate to the repository folder
cd adaptive-temporal-relational-ids

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run Phase 1 Controlled Experiments

```bash
# Execute specific dataset + loss experiment:
python scripts/run_phase1_experiments.py --dataset nf_cse_cic_ids2018_v2 --loss standard_ce

# Execute all 3 losses for a given dataset:
python scripts/run_phase1_experiments.py --dataset nf_unsw_nb15_v2 --loss all --epochs 15

# Execute all 9 controlled experiments:
python scripts/run_phase1_experiments.py --dataset all --loss all --epochs 15
```

### 3. Generate Master Summary Report

```bash
python scripts/generate_phase1_report.py results/phase1_loss_experiments
```

Outputs:
- `results/phase1_loss_experiments/phase1_summary/master_results_table.csv`
- `results/phase1_loss_experiments/phase1_summary/PHASE_1_EXPERIMENTAL_REPORT.md`

---

## 📊 Candidate Dataset Roles & Stratification

| Dataset Identifier | Tier | Role in Research Framework | Format / Schema | Est. Records | Temporal-Relational Readiness |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **NF-CSE-CIC-IDS2018-v2** | **1** | Core Enterprise NetFlow Benchmark | NetFlow v9 (43 feats) | ~18.8M | **High (IP, Port, Proto, Epoch Time)** |
| **NF-UNSW-NB15-v2** | **1** | Core Mixed Enterprise/Cyber-Range NetFlow | NetFlow v9 (43 feats) | ~2.3M | **High (IP, Port, Proto, Epoch Time)** |
| **CSE-CIC-IDS2018** | **1** | Deep Bidirectional Flow Richness | CICFlowMeter v3 (80 feats) | ~16.2M | **High (Subflow stats, IAT, Windows)** |
| **NF-ToN-IoT-v2** | **2** | Zero-Shot Cross-Domain IoT NetFlow | NetFlow v9 (43 feats) | ~16.9M | **High (Standardized NetFlow on IoT)** |
| **LYCOS-IDS2017** | **2** | Quality-Corrected Sensitivity Twin of CIC2017 | LycoSTand (83 feats) | ~2.8M | **High (Corrected TCP teardown/IAT)** |
| **CIC-IDS2017** | **2** | Extractor Baseline (CICFlowMeter v1) | CICFlowMeter v1 (79 feats) | ~2.8M | **Moderate (Known extraction artifacts)** |
| **UNSW-NB15** | **2** | Cyber-Range Deep Flow Baseline | Argus/Bro (49 feats) | ~2.5M | **High (Attack categories & timestamps)** |
| **CICIoT2023** | **2** | Contemporary IoT Multi-Device Benchmark | CICFlowMeter (47 feats) | ~46.0M | **High (33 devices, 105 attack classes)** |
| **InSDN** | **3** | SDN Virtual Network Reference | SDN Flow (83 feats) | ~340K | **Moderate (SDN-specific control flows)** |
| **NSL-KDD** | **3** | Legacy Reference Baseline | KDD Vector (41 feats) | ~148K | **Excluded from Dynamic Graph Fusion** |
