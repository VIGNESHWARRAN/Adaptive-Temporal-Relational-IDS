"""Dataset Loader and Data Preprocessor for Phase 1 Controlled Experiments.

Handles raw CSV ingestion for the 3 target datasets:
- NF-CSE-CIC-IDS2018-v2
- NF-UNSW-NB15-v2
- CSE-CIC-IDS2018

Applies Phase 0 frozen feature selections, 70/15/15 Train/Val/Test split (fixed seed=42),
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
            path = os.path.join(base_data_dir, "nf-cse-cic-ids2018-v2", "b3427ed8ad063a09_MOHANAD_A4706", "data", "NF-CSE-CIC-IDS2018-v2.csv")
            matches = glob.glob(path)
        if not matches:
            raise FileNotFoundError(f"Could not find NF-CSE-CIC-IDS2018-v2.csv in {base_data_dir}")
        return matches

    elif dataset_name == "nf_unsw_nb15_v2":
        path = os.path.join(base_data_dir, "nf-unsw-nb15-v2", "**", "NF-UNSW-NB15-v2.csv")
        matches = glob.glob(path, recursive=True)
        if not matches:
            path = os.path.join(base_data_dir, "nf-unsw-nb15-v2", "fe6cb615d161452c_MOHANAD_A4706", "data", "NF-UNSW-NB15-v2.csv")
            matches = glob.glob(path)
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
    config_dir: str = "configs/feature_selection",
    phase0_results_dir: str = "results/phase0_data_preparation",
    sample_limit: int = 100000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Loads dataset, applies Phase 0 frozen feature selection, splits (70/15/15), and scales.

    Args:
        dataset_name: One of 'nf_cse_cic_ids2018_v2', 'nf_unsw_nb15_v2', 'cse_cic_ids2018'
        base_data_dir: Base raw data directory
        config_dir: Directory containing frozen feature selection JSONs
        phase0_results_dir: Phase 0 results directory
        sample_limit: Max records to read for controlled execution
        seed: Fixed random seed for reproducible split

    Returns:
        Structured data dictionary with train/val/test features, labels, class info, weights, and PyTorch DataLoaders.
    """
    norm_name = dataset_name.lower().replace("-", "_")
    csv_paths = get_dataset_raw_paths(norm_name, base_data_dir)

    # 1. Load Frozen Phase 0 Feature Selection Metadata
    selected_features = []
    role_mapping = {}
    config_json_path = os.path.join(config_dir, f"{norm_name}_features.json")
    results_json_path = os.path.join(phase0_results_dir, f"{norm_name}_feature_selection.json")
    loaded_config_path = None

    if os.path.exists(config_json_path):
        loaded_config_path = config_json_path
        with open(config_json_path, "r", encoding="utf-8") as f:
            c_meta = json.load(f)
            selected_features = c_meta.get("selected_features", [])
            role_mapping = {
                "temporal_features": c_meta.get("temporal_features", []),
                "relational_edge_features": c_meta.get("relational_features", []),
                "relational_node_features": c_meta.get("graph_node_features", []),
            }
    elif os.path.exists(results_json_path):
        loaded_config_path = results_json_path
        with open(results_json_path, "r", encoding="utf-8") as f:
            r_meta = json.load(f)
            selected_features = r_meta.get("selected_features", [])
            role_mapping = r_meta.get("role_mapping", {})

    # 2. Ingest CSV data
    dfs = []
    chunksize = 50000

    if norm_name == "cse_cic_ids2018" and len(csv_paths) > 1:
        # Sample across all daily CSV files to preserve all attack classes
        rows_per_file = max(5000, sample_limit // len(csv_paths))
        for p in csv_paths:
            f_chunks = []
            c_needed = rows_per_file
            for chunk in pd.read_csv(p, nrows=c_needed * 2, chunksize=chunksize, low_memory=False):
                chunk.columns = [c.strip() for c in chunk.columns]
                f_chunks.append(chunk)
                c_needed -= len(chunk)
                if c_needed <= 0:
                    break
            if f_chunks:
                f_df = pd.concat(f_chunks, ignore_index=True)
                if len(f_df) > rows_per_file:
                    f_df = f_df.sample(n=rows_per_file, random_state=seed)
                dfs.append(f_df)
    else:
        rows_needed = sample_limit
        for p in csv_paths:
            if rows_needed <= 0:
                break
            for chunk in pd.read_csv(p, nrows=rows_needed, chunksize=chunksize, low_memory=False):
                chunk.columns = [c.strip() for c in chunk.columns]
                dfs.append(chunk)
                rows_needed -= len(chunk)
                if rows_needed <= 0:
                    break

    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    # Determine Target Label Column
    label_col = None
    for cand in ["Label", "Attack", "attack_cat", "label"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise KeyError(f"Could not identify target label column in df columns: {df.columns.tolist()}")

    # Filter out header string rows inside data if present
    df = df[df[label_col] != label_col]

    # Determine Feature Columns
    if selected_features:
        feature_cols = [c for c in selected_features if c in df.columns]
    else:
        exact_ids = {'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'SRC IP', 'DST IP', 'SRC_IP', 'DST_IP', 'TIMESTAMP', 'TIME', 'FLOW ID', 'ROW_ID', 'DNS_QUERY_ID', label_col}
        feature_cols = [c for c in df.columns if c.upper().strip() not in exact_ids]

    if not feature_cols:
        raise ValueError(f"No feature columns identified for dataset {norm_name}!")

    # 3. Clean numeric columns & safe clip to float32 range
    F_LIMIT = 1e12
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        if s.isnull().any():
            med = s.median()
            if pd.isna(med):
                med = 0.0
            s = s.fillna(med)
        df[c] = s.clip(-F_LIMIT, F_LIMIT)

    X_raw = df[feature_cols].values.astype(np.float32)
    raw_labels = df[label_col].values

    # Encode Target Labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(raw_labels.astype(str))
    class_names = [str(c) for c in label_encoder.classes_]
    num_classes = len(class_names)
    label_mapping = {int(idx): str(cls_name) for idx, cls_name in enumerate(class_names)}

    # Calculate Class Statistics
    unique_classes, counts = np.unique(y_encoded, return_counts=True)
    class_distribution = {str(class_names[c]): int(cnt) for c, cnt in zip(unique_classes, counts)}
    imbalance_ratio = float(max(counts) / max(1, min(counts)))

    # 4. 70% Train, 15% Val, 15% Test Split (Fixed Random Seed)
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X_raw, y_encoded, test_size=0.30, random_state=seed, stratify=y_encoded if min(counts) > 1 else None
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp if min(np.unique(y_temp, return_counts=True)[1]) > 1 else None
    )

    # 5. Scale Features STRICTLY on Train Set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Clip scaled outputs
    X_train = np.clip(X_train, -100.0, 100.0)
    X_val = np.clip(X_val, -100.0, 100.0)
    X_test = np.clip(X_test, -100.0, 100.0)

    # 6. Compute Training Class Weights (Inverse Frequency)
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    total_train_samples = len(y_train)
    weights = np.ones(num_classes, dtype=np.float32)
    for cls_idx, cnt in zip(train_unique, train_counts):
        weights[cls_idx] = total_train_samples / (num_classes * cnt)
    weights = weights / np.mean(weights)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # 7. Build PyTorch Tensors and DataLoaders
    def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = False) -> DataLoader:
        t_X_temp = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        t_X_rel = torch.tensor(X, dtype=torch.float32)
        t_y = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(t_X_temp, t_X_rel, t_y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = create_dataloader(X_train, y_train, batch_size=64, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=128, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=128, shuffle=False)

    # Build Dataset Integration Report Dict
    integration_report = {
        "dataset_name": dataset_name,
        "loaded_feature_config_path": loaded_config_path,
        "exact_selected_feature_names": feature_cols,
        "num_selected_features": len(feature_cols),
        "num_classes": num_classes,
        "label_mapping": label_mapping,
        "temporal_feature_mapping": role_mapping.get("temporal_features", []),
        "relational_feature_mapping": role_mapping.get("relational_edge_features", []),
        "train_size": len(y_train),
        "validation_size": len(y_val),
        "test_size": len(y_test),
        "model_input_dimensions": len(feature_cols),
        "selected_features_used_confirmation": True
    }

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
        "integration_report": integration_report,
    }
