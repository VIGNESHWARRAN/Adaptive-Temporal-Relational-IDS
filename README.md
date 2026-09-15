# 🛡️ Adaptive Temporal-Relational Intrusion Detection System (IDS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Colab / Local](https://img.shields.io/badge/platform-Google%20Colab%20%7C%20Local-orange.svg)]()
[![License: Academic Research](https://img.shields.io/badge/license-Academic%20Research-green.svg)]()

> **Project:** *Adaptive Temporal-Relational Intrusion Detection for Evolving Network Traffic: Investigating Classical, Quantum, and Continual Learning Approaches*

---

## 📖 Overview

Conventional intrusion detection systems treat network flows as independent tabular records, discarding critical **temporal sequences** (event ordering, inter-arrival times, multi-stage burst dynamics) and **relational topology** (communication graphs $G_t = (V_t, E_t)$, entity interactions, host connectivity).

This repository provides the core data engineering and pipeline foundation:
1. **Multi-Source Dataset Ingestion Engine:** Resilient, chunked downloading supporting direct HTTP/HTTPS, Zenodo, and Kaggle API.
2. **Standardized NetFlow & Deep Flow Schema Alignment:** Supporting Tier 1 core enterprise NetFlow (`NF-CSE-CIC-IDS2018-v2`, `NF-UNSW-NB15-v2`), deep flow statistics (`CSE-CIC-IDS2018`), and cross-domain IoT benchmarks (`NF-ToN-IoT-v2`, `CICIoT2023`).
3. **Automated Metadata Profiler & Readiness Evaluator:** Inspects schemas, null rates, record counts, and scores datasets on **Temporal & Relational readiness** for Graph Neural Networks and Sequence Models.
4. **Zero-Laptop-Spec Google Colab Workflow:** Ready-to-run Colab integration with automatic Google Drive persistence.

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
│   ├── download/
│   │   ├── __init__.py
│   │   ├── dataset_registry.py         # Registry loader & tier filter
│   │   └── downloader.py               # Resumable HTTP, KaggleHub & Zenodo downloader
│   ├── metadata/
│   │   ├── __init__.py
│   │   ├── extractor.py                # Schema, null rate, and class distribution profiler
│   │   ├── temporal_relational_eval.py # Relational/Temporal feature compatibility evaluator
│   │   └── reporter.py                 # Generates Markdown and JSON metadata reports
│   └── utils/
│       ├── __init__.py
│       └── colab_utils.py              # Environment diagnostics & Google Drive mount helper
├── scripts/
│   ├── download_datasets.py            # CLI script to download datasets
│   ├── extract_metadata.py             # CLI script to profile datasets and create reports
│   └── run_all_pipeline.py             # Master pipeline orchestrator
├── notebooks/
│   └── 01_download_and_metadata_colab.ipynb  # Interactive 1-click Colab notebook
└── metadata_reports/                   # Output folder for generated metadata JSON & MD reports
```

---

## 🚀 Quickstart: Running on Google Colab (Recommended)

Since model training and large dataset preprocessing require compute and storage, use **Google Colab**:

1. Open **[Google Colab](https://colab.research.google.com/)**.
2. Upload and open [`notebooks/01_download_and_metadata_colab.ipynb`](notebooks/01_download_and_metadata_colab.ipynb).
3. Execute the cells in order:
   - **Cell 1 & 2:** Mounts your Google Drive at `/content/drive/MyDrive/IDS_Research_Project`.
   - **Cell 3:** Installs lightweight dependencies.
   - **Cell 4:** Downloads Tier 1 NetFlow datasets directly to Google Drive.
   - **Cell 5 & 6:** Runs metadata extraction and previews the generated Markdown reports.

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

### 2. Check Dataset Registry (Dry Run)

```bash
# Simulate download plan for Tier 1 datasets without fetching large files
python scripts/download_datasets.py --tier 1 --dry-run
```

### 3. Download Datasets

```bash
# Download Tier 1 Core Primary NetFlow datasets (NF-CSE-CIC-IDS2018-v2 & NF-UNSW-NB15-v2)
python scripts/download_datasets.py --tier 1

# Or download specific datasets:
python scripts/download_datasets.py --datasets nf-unsw-nb15-v2 nf-cse-cic-ids2018-v2
```

### 4. Extract Metadata & Generate Reports

```bash
# Scan data/raw/ and generate Markdown & JSON reports in metadata_reports/
python scripts/extract_metadata.py
```

### 5. Run Entire Ingestion & Profiling Pipeline

```bash
python scripts/run_all_pipeline.py
```

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

---

## 🔧 Git Initialization & Setup

To initialize this folder as a git repository and connect it to your GitHub / GitLab:

```bash
cd adaptive-temporal-relational-ids
git init
git add .
git commit -m "feat: initial commit with dataset downloaders, metadata profilers, and Colab pipeline"
git branch -M main
# Add your remote repository:
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```
