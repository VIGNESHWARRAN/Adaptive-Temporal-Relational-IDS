"""
Feature Selection Validator Module for Phase 0 (Step 21 & Step 25).

Performs lightweight baseline validation on TRAIN / VALIDATION split:
Compares 'All Cleaned Candidate Features' vs 'Selected Feature Subset'.
Verifies feature selection does NOT cause significant loss of predictive information.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

class FeatureValidator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def validate_selection(
        self,
        dataset_name: str,
        df: pd.DataFrame,
        selected_features: List[str],
        target_col: str = 'Label',
        attack_col: str = None
    ) -> Dict[str, Any]:
        print(f"\n--> Validating Feature Selection for {dataset_name} on Train/Val Split...")

        # 1. Clean data & encode target
        df_clean = df.copy()
        df_clean.columns = [c.strip() for c in df_clean.columns]
        
        # Convert object columns representing numbers to numeric dtypes
        for c in df_clean.columns:
            if c not in [target_col, attack_col] and c.upper().strip() not in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'SRC IP', 'DST IP', 'SRC_IP', 'DST_IP', 'TIMESTAMP', 'TIME', 'FLOW ID', 'ROW_ID', 'DNS_QUERY_ID']:
                if df_clean[c].dtype == 'object':
                    converted = pd.to_numeric(df_clean[c], errors='coerce')
                    if converted.notnull().sum() > 0:
                        df_clean[c] = converted

        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        F_LIMIT = 1e12
        for c in num_cols:
            s = df_clean[c].replace([np.inf, -np.inf], np.nan)
            if s.isnull().any():
                med = s.median()
                if pd.isna(med):
                    med = 0.0
                s = s.fillna(med)
            df_clean[c] = s.clip(-F_LIMIT, F_LIMIT)

        le = LabelEncoder()
        y = le.fit_transform(df_clean[target_col].astype(str))

        ignored_cols = [c for c in [target_col, attack_col] if c and c in df_clean.columns]
        candidate_cols = [c for c in df_clean.columns if c not in ignored_cols]

        exact_id_names = {'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'SRC IP', 'DST IP', 'SRC_IP', 'DST_IP', 'TIMESTAMP', 'TIME', 'FLOW ID', 'ROW_ID', 'DNS_QUERY_ID'}
        pure_ids = [c for c in candidate_cols if c.upper().strip() in exact_id_names]
        all_numeric_cols = [c for c in candidate_cols if c not in pure_ids and np.issubdtype(df_clean[c].dtype, np.number)]

        selected_numeric_cols = [c for c in selected_features if c in all_numeric_cols]

        X_full = df_clean[all_numeric_cols].copy()
        X_selected = df_clean[selected_numeric_cols].copy()

        # 2. Train/Val split (70% Train, 30% Val)
        X_full_tr, X_full_val, y_tr, y_val = train_test_split(X_full, y, test_size=0.30, random_state=42, stratify=y)
        X_sel_tr, X_sel_val, _, _ = train_test_split(X_selected, y, test_size=0.30, random_state=42, stratify=y)

        # 3. Evaluate baseline LightGBM model on Full vs Selected
        clf_full = LGBMClassifier(n_estimators=50, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
        clf_full.fit(X_full_tr, y_tr)
        y_pred_full = clf_full.predict(X_full_val)

        full_metrics = {
            'accuracy': round(float(accuracy_score(y_val, y_pred_full)), 6),
            'macro_f1': round(float(f1_score(y_val, y_pred_full, average='macro')), 6),
            'weighted_f1': round(float(f1_score(y_val, y_pred_full, average='weighted')), 6),
            'precision': round(float(precision_score(y_val, y_pred_full, average='weighted')), 6),
            'recall': round(float(recall_score(y_val, y_pred_full, average='weighted')), 6)
        }

        clf_sel = LGBMClassifier(n_estimators=50, max_depth=6, random_state=42, verbose=-1, n_jobs=-1)
        clf_sel.fit(X_sel_tr, y_tr)
        y_pred_sel = clf_sel.predict(X_sel_val)

        selected_metrics = {
            'accuracy': round(float(accuracy_score(y_val, y_pred_sel)), 6),
            'macro_f1': round(float(f1_score(y_val, y_pred_sel, average='macro')), 6),
            'weighted_f1': round(float(f1_score(y_val, y_pred_sel, average='weighted')), 6),
            'precision': round(float(precision_score(y_val, y_pred_sel, average='weighted')), 6),
            'recall': round(float(recall_score(y_val, y_pred_sel, average='weighted')), 6)
        }

        f1_diff = selected_metrics['macro_f1'] - full_metrics['macro_f1']
        retention = (selected_metrics['macro_f1'] / max(full_metrics['macro_f1'], 1e-6)) * 100.0

        results = {
            'dataset_name': dataset_name,
            'full_candidate_features_count': len(all_numeric_cols),
            'selected_features_count': len(selected_numeric_cols),
            'full_feature_metrics': full_metrics,
            'selected_feature_metrics': selected_metrics,
            'macro_f1_difference': round(float(f1_diff), 6),
            'performance_retention_pct': round(float(retention), 4),
            'validation_pass': bool(retention >= 98.0)
        }

        print(f"--> Full Feature Macro F1: {full_metrics['macro_f1']} | Selected Feature Macro F1: {selected_metrics['macro_f1']}")
        print(f"--> Performance Retention: {round(retention, 2)}% | Validation Passed: {results['validation_pass']}")

        out_path = os.path.join(self.output_dir, f"{dataset_name}_validation.json")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results
