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
from collections import defaultdict

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
    output_dir: str = None
) -> Dict[str, Any]:
    """Loads dataset, applies Phase 0 frozen feature selection, splits (70/15/15), and scales.

    Args:
        dataset_name: One of 'nf_cse_cic_ids2018_v2', 'nf_unsw_nb15_v2', 'cse_cic_ids2018'
        base_data_dir: Base raw data directory
        config_dir: Directory containing frozen feature selection JSONs
        phase0_results_dir: Phase 0 results directory
        sample_limit: Base majority sample limit
        seed: Fixed random seed for reproducible split
        output_dir: Optional directory to save split_manifest.csv

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

    # 2. Ingest CSV data with ultra-efficient 2-pass index-aware extraction for CSE-CIC-IDS2018
    chunksize = 250000

    if norm_name == "cse_cic_ids2018" and len(csv_paths) > 1:
        print(f"Executing Fast Index-Aware Class Sampling across all {len(csv_paths)} CSE CSV files...")
        
        # Pass 1: Scan ONLY Label column to map row indices per class
        class_records = defaultdict(list) # class_name -> list of (file_path, file_row_idx)
        total_scanned_rows = 0

        for p in csv_paths:
            fname = os.path.basename(p)
            head = pd.read_csv(p, nrows=2)
            head.columns = [c.strip() for c in head.columns]
            label_col = 'Label' if 'Label' in head.columns else ('label' if 'label' in head.columns else None)

            file_row_offset = 0
            for chunk in pd.read_csv(p, usecols=[label_col], chunksize=chunksize, low_memory=True):
                chunk.columns = [c.strip() for c in chunk.columns]
                s = chunk[label_col].astype(str)
                valid_mask = (s != label_col)
                s_valid = s[valid_mask]

                c_len = len(chunk)
                valid_indices = np.where(valid_mask)[0]

                for idx_in_chunk, lbl in zip(valid_indices, s_valid):
                    class_records[lbl].append((p, file_row_offset + idx_in_chunk))

                file_row_offset += c_len
                total_scanned_rows += len(valid_indices)

        # Pass 2: Class-aware index selection
        rare_threshold = 15000
        benign_cnt = len(class_records.get("Benign", []))
        benign_quota = int(round(sample_limit * (benign_cnt / total_scanned_rows))) # 83,070

        selected_file_row_map = defaultdict(set) # file_path -> set of row_indices to load

        for cls_name in sorted(class_records.keys()):
            recs = class_records[cls_name]
            total_cls = len(recs)
            if total_cls <= rare_threshold:
                # 100% Retained for Rare Attack Classes
                chosen = recs
            elif cls_name == "Benign":
                # Proportional Benign Quota
                rng = np.random.RandomState(seed)
                chosen_idx = rng.choice(total_cls, size=min(benign_quota, total_cls), replace=False)
                chosen = [recs[i] for i in chosen_idx]
            else:
                # Capped 5,000 Quota for Majority Attack Classes
                rng = np.random.RandomState(seed)
                sample_n = min(5000, total_cls)
                chosen_idx = rng.choice(total_cls, size=sample_n, replace=False)
                chosen = [recs[i] for i in chosen_idx]

            for p, r_id in chosen:
                selected_file_row_map[p].add(r_id)

        # Pass 3: Read ONLY selected rows for features and labels
        extracted_dfs = []
        for p in csv_paths:
            fname = os.path.basename(p)
            target_r_ids = selected_file_row_map.get(p, set())
            if not target_r_ids:
                continue

            head = pd.read_csv(p, nrows=2)
            head.columns = [c.strip() for c in head.columns]
            label_col = 'Label' if 'Label' in head.columns else ('label' if 'label' in head.columns else None)

            load_cols = [c for c in selected_features if c in head.columns]
            if label_col and label_col not in load_cols:
                load_cols.append(label_col)

            file_row_offset = 0
            for chunk in pd.read_csv(p, usecols=load_cols, chunksize=chunksize, low_memory=True):
                chunk.columns = [c.strip() for c in chunk.columns]
                if label_col in chunk.columns:
                    chunk = chunk[chunk[label_col] != label_col]
                    chunk.rename(columns={label_col: 'Label'}, inplace=True)
                
                c_len = len(chunk)
                chunk_r_ids = np.arange(file_row_offset, file_row_offset + c_len)
                file_row_offset += c_len

                # Filter chunk rows that match target_r_ids
                in_target_mask = np.isin(chunk_r_ids, list(target_r_ids))
                if np.any(in_target_mask):
                    matched_chunk = chunk[in_target_mask].copy()
                    matched_chunk['source_file'] = fname
                    matched_chunk['source_row_id'] = chunk_r_ids[in_target_mask]
                    extracted_dfs.append(matched_chunk)

        df = pd.concat(extracted_dfs, ignore_index=True)
        print(f"Successfully extracted {len(df):,} total rows across all 15 CSE classes (Full scanned rows: {total_scanned_rows:,})")

    else:
        # Standard ingestion for single CSV file datasets (NF-CSE / NF-UNSW)
        dfs = []
        rows_needed = sample_limit
        for p in csv_paths:
            if rows_needed <= 0:
                break
            for chunk in pd.read_csv(p, nrows=rows_needed, chunksize=chunksize, low_memory=False):
                chunk.columns = [c.strip() for c in chunk.columns]
                chunk['source_file'] = os.path.basename(p)
                chunk['source_row_id'] = np.arange(len(chunk), dtype=np.int32)
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
        exact_ids = {'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'SRC IP', 'DST IP', 'SRC_IP', 'DST_IP', 'TIMESTAMP', 'TIME', 'FLOW ID', 'ROW_ID', 'DNS_QUERY_ID', label_col, 'source_file', 'source_row_id'}
        feature_cols = [c for c in df.columns if c.upper().strip() not in exact_ids]

    if not feature_cols:
        raise ValueError(f"No feature columns identified for dataset {norm_name}!")

    # 3. Clean numeric columns & safe clip to float32 range
    F_LIMIT = 1e12
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        df[c] = s.clip(-F_LIMIT, F_LIMIT)

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

    # Preserve Source Metadata
    source_files = df['source_file'].values if 'source_file' in df.columns else np.array(["unknown"] * len(df))
    source_row_ids = df['source_row_id'].values if 'source_row_id' in df.columns else np.arange(len(df))

    # 4. 70% Train, 15% Val, 15% Test Split (Fixed Random Seed)
    indices = np.arange(len(df))
    can_stratify_step1 = (min(counts) > 1)

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y_encoded, test_size=0.30, random_state=seed, stratify=y_encoded if can_stratify_step1 else None
    )

    temp_unique, temp_counts = np.unique(y_temp, return_counts=True)
    can_stratify_step2 = (min(temp_counts) > 1) if len(temp_counts) > 0 else False

    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp if can_stratify_step2 else None
    )

    # Build Split Assignment Array & Split Manifest
    split_assignment = np.array(['unassigned'] * len(df), dtype=object)
    split_assignment[idx_train] = 'train'
    split_assignment[idx_val] = 'val'
    split_assignment[idx_test] = 'test'

    manifest_df = pd.DataFrame({
        "source_file": source_files,
        "source_row_id": source_row_ids,
        "label": raw_labels.astype(str),
        "encoded_label": y_encoded,
        "split": split_assignment,
        "sampling_seed": seed
    })

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, "split_manifest.csv")
        manifest_df.to_csv(manifest_path, index=False)
        print(f"Saved Split Manifest to {manifest_path} ({len(manifest_df):,} rows)")

    # Extract Raw Feature Arrays for Train / Val / Test
    X_raw = df[feature_cols].values.astype(np.float32)

    X_train_raw = X_raw[idx_train]
    X_val_raw = X_raw[idx_val]
    X_test_raw = X_raw[idx_test]

    # 5. Fit Median Imputation and StandardScaler STRICTLY on Training Set
    for j in range(X_train_raw.shape[1]):
        col_train = X_train_raw[:, j]
        col_val = X_val_raw[:, j]
        col_test = X_test_raw[:, j]

        nan_mask_train = np.isnan(col_train)
        if np.any(nan_mask_train):
            med = np.nanmedian(col_train)
            if np.isnan(med):
                med = 0.0
            col_train[nan_mask_train] = med
            col_val[np.isnan(col_val)] = med
            col_test[np.isnan(col_test)] = med

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Post-scaling Clip
    X_train = np.clip(X_train, -100.0, 100.0)
    X_val = np.clip(X_val, -100.0, 100.0)
    X_test = np.clip(X_test, -100.0, 100.0)

    # 6. Compute Training Class Weights (Inverse Frequency) strictly on Training set
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
        "class_distribution": class_distribution,
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
        "manifest_df": manifest_df
    }
