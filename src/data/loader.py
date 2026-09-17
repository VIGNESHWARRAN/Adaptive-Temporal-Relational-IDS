"""Dataset Loader and Data Preprocessor for Phase 1 Controlled Experiments.

Handles raw CSV ingestion for the 3 target datasets:
- NF-CSE-CIC-IDS2018-v2
- NF-UNSW-NB15-v2
- CSE-CIC-IDS2018

Applies Phase 0 feature selections, 70/15/15 Train/Val/Test split (fixed seed=42),
fits StandardScaler strictly on training set, and computes class weights.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def get_dataset_raw_paths(dataset_name: str, base_data_dir: str = "data/raw") -> List[str]:
    """Resolves raw CSV file paths for the given dataset."""
    dataset_name = dataset_name.lower().replace("-", "_")

    if dataset_name == "nf_cse_cic_ids2018_v2":
        path = os.path.join(base_data_dir, "nf-cse-cic-ids2018-v2", "**", "NF-CSE-CIC-IDS2018-v2.csv")
        matches = glob.glob(path, recursive=True)
        if not matches:
            raise FileNotFoundError(f"Could not find NF-CSE-CIC-IDS2018-v2.csv in {base_data_dir}")
        return matches

    elif dataset_name == "nf_unsw_nb15_v2":
        path = os.path.join(base_data_dir, "nf-unsw-nb15-v2", "**", "NF-UNSW-NB15-v2.csv")
        matches = glob.glob(path, recursive=True)
        if not matches:
            raise FileNotFoundError(f"Could not find NF-UNSW-NB15-v2.csv in {base_data_dir}")
        return matches

    elif dataset_name == "cse_cic_ids2018":
        path = os.path.join(base_data_dir, "cse-cic-ids2018", "*.csv")
        matches = glob.glob(path)
        if not matches:
            raise FileNotFoundError(f"Could not find cse-cic-ids2018 CSV files in {base_data_dir}")
        return sorted(matches)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def load_dataset_split(
    dataset_name: str,
    base_data_dir: str = "data/raw",
    phase0_results_dir: str = "results/results/phase0_data_preparation",
    sample_limit: int = 100000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Loads dataset, applies Phase 0 feature selection, splits (70/15/15), and scales.
    
    Args:
        dataset_name: One of 'nf_cse_cic_ids2018_v2', 'nf_unsw_nb15_v2', 'cse_cic_ids2018'
        base_data_dir: Base raw data directory
        phase0_results_dir: Phase 0 results directory containing feature selection JSONs
        sample_limit: Max records to read for controlled execution
        seed: Fixed random seed for reproducible split
        
    Returns:
        Structured data dictionary with train/val/test features, labels, class info, weights, and PyTorch DataLoaders.
    """
    norm_name = dataset_name.lower().replace("-", "_")
    csv_paths = get_dataset_raw_paths(norm_name, base_data_dir)

    # 1. Load Phase 0 Feature Selection Metadata
    fs_json_path = os.path.join(phase0_results_dir, f"{norm_name}_feature_selection.json")
    role_mapping = None
    if os.path.exists(fs_json_path):
        with open(fs_json_path, "r") as f:
            fs_meta = json.load(f)
            role_mapping = fs_meta.get("role_mapping", {})

    # 2. Ingest CSV data
    dfs = []
    rows_needed = sample_limit
    for p in csv_paths:
        if rows_needed <= 0:
            break
        # Read chunk or sample
        chunk = pd.read_csv(p, nrows=rows_needed)
        dfs.append(chunk)
        rows_needed -= len(chunk)

    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    # Clean missing / inf values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Determine Target Label Column
    label_col = None
    for cand in ["Label", "Attack", "attack_cat", "label"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise KeyError(f"Could not identify target label column in df columns: {df.columns.tolist()}")

    # Determine Feature Columns
    if role_mapping:
        temp_cols = [c for c in role_mapping.get("temporal_features", []) if c in df.columns]
        edge_cols = [c for c in role_mapping.get("relational_edge_features", []) if c in df.columns]
        beh_cols = [c for c in role_mapping.get("behavioral_features", []) if c in df.columns]
        feature_cols = list(dict.fromkeys(temp_cols + edge_cols + beh_cols))
    else:
        # Fallback to numerical features excluding label and pure identifiers
        exclude_cols = [label_col, "Timestamp", "Flow ID", "Src IP", "Dst IP", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "DNS_QUERY_ID"]
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

    if not feature_cols:
        raise ValueError("No feature columns identified for dataset ingestion!")

    X_raw = df[feature_cols].values.astype(np.float32)
    raw_labels = df[label_col].values

    # Encode Target Labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(raw_labels)
    class_names = [str(c) for c in label_encoder.classes_]
    num_classes = len(class_names)

    # Calculate Class Statistics
    unique_classes, counts = np.unique(y_encoded, return_counts=True)
    class_distribution = {str(class_names[c]): int(cnt) for c, cnt in zip(unique_classes, counts)}
    imbalance_ratio = float(max(counts) / max(1, min(counts)))

    # 3. 70% Train, 15% Val, 15% Test Split (Fixed Random Seed)
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_encoded, test_size=0.30, random_state=seed, stratify=y_encoded if min(counts) > 1 else None
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp if min(np.unique(y_temp, return_counts=True)[1]) > 1 else None
    )

    # 4. Scale Features STRICTLY on Train Set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # 5. Compute Training Class Weights (Inverse Frequency)
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    total_train_samples = len(y_train)
    weights = np.ones(num_classes, dtype=np.float32)
    for cls_idx, cnt in zip(train_unique, train_counts):
        weights[cls_idx] = total_train_samples / (num_classes * cnt)
    # Normalize weights so mean weight = 1.0
    weights = weights / np.mean(weights)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # 6. Build PyTorch Tensors and DataLoaders
    # Temporal part: X_train (batch, 1, feature_dim) or (batch, feature_dim)
    # Relational node part: X_train (batch, feature_dim)
    def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
        t_X_temp = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (batch, 1, input_dim)
        t_X_rel = torch.tensor(X, dtype=torch.float32)                # (batch, input_dim)
        t_y = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(t_X_temp, t_X_rel, t_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = create_dataloader(X_train, y_train, batch_size=64, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=128, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=128, shuffle=False)

    return {
        "dataset_name": dataset_name,
        "feature_names": feature_cols,
        "class_names": class_names,
        "num_classes": num_classes,
        "class_distribution": class_distribution,
        "imbalance_ratio": imbalance_ratio,
        "input_dim": len(feature_cols),
        "class_weights": class_weights_tensor,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler_fitted_on_train_only": True,
    }
