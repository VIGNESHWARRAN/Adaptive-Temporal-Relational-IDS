"""
Feature Selector Module for Phase 0.

Strict Rule Enforcement:
- ALL feature selection calculations (MI, ANOVA, Random Forest / XGBoost importance, Correlation Analysis)
  use the TRAIN split ONLY (70% stratified split).
- The test split is NEVER accessed or used during feature selection.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

class FeatureSelector:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_dataset(
        self,
        dataset_name: str,
        df: pd.DataFrame,
        target_col: str = 'Label',
        attack_col: str = None,
        exclude_cols: List[str] = None
    ) -> Dict[str, Any]:
        print(f"\n=======================================================")
        print(f"--> Multi-Signal Feature Selection for {dataset_name}")
        print(f"=======================================================")

        if exclude_cols is None:
            exclude_cols = []

        # 1. Clean Data & Encode Target
        df_clean, target_series, label_encoder = self._clean_and_prep(df, target_col, attack_col)

        # Explicitly separate candidate feature matrix X and target y
        ignored_cols = [c for c in [target_col, attack_col] if c and c in df_clean.columns] + exclude_cols
        candidate_cols = [c for c in df_clean.columns if c not in ignored_cols]

        X_full = df_clean[candidate_cols].copy()
        y_full = target_series.values

        # Remove non-numeric or raw identifier columns from candidate feature matrix for statistical modeling
        # (IP addresses, Timestamps, Row IDs are handled separately as graph/sequence attributes)
        exact_id_names = {'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'SRC IP', 'DST IP', 'SRC_IP', 'DST_IP', 'TIMESTAMP', 'TIME', 'FLOW ID', 'ROW_ID', 'DNS_QUERY_ID'}
        pure_ids = [c for c in candidate_cols if c.upper().strip() in exact_id_names]
        numeric_candidate_cols = [c for c in candidate_cols if c not in pure_ids and np.issubdtype(X_full[c].dtype, np.number)]

        print(f"Total rows: {len(df_clean)} | Total candidates: {len(candidate_cols)} | Numeric tabular candidates: {len(numeric_candidate_cols)}")
        print(f"Excluded identifier/timestamp fields from tabular matrix X: {pure_ids}")

        X_num = X_full[numeric_candidate_cols].copy()

        # 2. STRICT 70/15/15 TRAIN/VAL/TEST SPLIT (Feature selection uses TRAIN ONLY)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_num, y_full, test_size=0.30, random_state=42, stratify=y_full
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        print(f"--> Split sizes — Train: {len(X_train)} (70%), Val: {len(X_val)} (15%), Test: {len(X_test)} (15%)")
        print(f"--> STRICT RULE ENFORCED: Feature selection relies 100% on TRAIN split.")

        # 3. Constant and Near-Constant Filter (on TRAIN)
        constant_cols, near_constant_cols = self._detect_constant_features(X_train)
        print(f"Constant features (variance=0): {constant_cols}")
        print(f"Near-constant features (>99.99% single value): {near_constant_cols}")

        active_cols = [c for c in numeric_candidate_cols if c not in constant_cols and c not in near_constant_cols]
        X_train_active = X_train[active_cols]

        # 4. Correlation & Redundancy Analysis (on TRAIN)
        corr_matrix, high_corr_pairs, redundancy_groups = self._analyze_correlation(X_train_active)

        # 5. Target-Related Feature Relevance (MI & ANOVA on TRAIN)
        mi_scores, mi_ranks = self._compute_mutual_info(X_train_active, y_train)
        f_scores, f_ranks = self._compute_anova_f(X_train_active, y_train)

        # 6. Model-Based Feature Importance (Random Forest & XGBoost on TRAIN)
        rf_importances, rf_ranks, perm_importances = self._compute_rf_importance(X_train_active, y_train)
        xgb_importances, xgb_ranks = self._compute_xgb_importance(X_train_active, y_train)

        # 7. Multi-Signal Evidence Table Assembly
        evidence_table = self._build_evidence_table(
            active_cols,
            mi_scores, mi_ranks,
            f_scores, f_ranks,
            rf_importances, rf_ranks,
            xgb_importances, xgb_ranks,
            high_corr_pairs,
            dataset_name
        )

        # 8. Selection Decision & Role Assignment
        selected_features, dropped_features, role_mapping = self._make_selection_decisions(
            evidence_table,
            dataset_name,
            pure_ids
        )

        print(f"--> FINAL DECISION: Selected {len(selected_features)} features out of {len(candidate_cols)} initial candidates.")
        print(f"--> Selected features: {selected_features}")

        results = {
            'dataset_name': dataset_name,
            'total_initial_candidates': len(candidate_cols),
            'excluded_identifiers_timestamps': pure_ids,
            'constant_features_removed': constant_cols,
            'near_constant_features_removed': near_constant_cols,
            'high_correlation_pairs': high_corr_pairs,
            'redundancy_groups': redundancy_groups,
            'evidence_table': evidence_table,
            'selected_features': selected_features,
            'dropped_features': dropped_features,
            'role_mapping': role_mapping,
            'splits_info': {
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test)
            }
        }

        # Save results JSON
        out_path = os.path.join(self.output_dir, f"{dataset_name}_feature_selection.json")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _clean_and_prep(self, df: pd.DataFrame, target_col: str, attack_col: str = None) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
        df = df.copy()
        # Clean column headers
        df.columns = [c.strip() for c in df.columns]

        # Convert object columns representing numbers to numeric dtypes
        for c in df.columns:
            if c not in [target_col, attack_col] and c.upper().strip() not in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'SRC IP', 'DST IP', 'SRC_IP', 'DST_IP', 'TIMESTAMP', 'TIME', 'FLOW ID', 'ROW_ID', 'DNS_QUERY_ID']:
                if df[c].dtype == 'object':
                    converted = pd.to_numeric(df[c], errors='coerce')
                    if converted.notnull().sum() > 0:
                        df[c] = converted

        # Handle numeric inf/NaN and clip to safe range [-1e12, 1e12]
        # (Prevents float32 overflow and square overflow in sklearn stats/tree fits)
        num_cols = df.select_dtypes(include=[np.number]).columns
        F_LIMIT = 1e12
        for c in num_cols:
            s = df[c].replace([np.inf, -np.inf], np.nan)
            if s.isnull().any():
                med = s.median()
                if pd.isna(med):
                    med = 0.0
                s = s.fillna(med)
            df[c] = s.clip(-F_LIMIT, F_LIMIT)

        # Encode target label cleanly
        le = LabelEncoder()
        target_series = pd.Series(le.fit_transform(df[target_col].astype(str)), index=df.index)

        return df, target_series, le

    def _detect_constant_features(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        constant_cols = []
        near_constant_cols = []
        n_rows = len(X)

        for col in X.columns:
            series = X[col]
            unq = series.nunique()
            if unq <= 1:
                constant_cols.append(col)
            else:
                top_freq = series.value_counts(dropna=False).iloc[0]
                ratio = top_freq / n_rows
                if ratio >= 0.9999:
                    near_constant_cols.append(col)

        return constant_cols, near_constant_cols

    def _analyze_correlation(self, X: pd.DataFrame, threshold: float = 0.90) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]], List[List[str]]]:
        corr_matrix = X.corr(method='pearson').fillna(0.0)
        high_corr_pairs = []
        cols = X.columns.tolist()

        visited = set()
        clusters = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                val = float(corr_matrix.loc[col1, col2])
                if abs(val) >= threshold:
                    high_corr_pairs.append({
                        'feature_a': col1,
                        'feature_b': col2,
                        'correlation': round(val, 4)
                    })

        # Build correlation clusters
        for col in cols:
            if col not in visited:
                cluster = [col]
                visited.add(col)
                for other in cols:
                    if other not in visited and abs(corr_matrix.loc[col, other]) >= threshold:
                        cluster.append(other)
                        visited.add(other)
                if len(cluster) > 1:
                    clusters.append(cluster)

        corr_dict = {col: {c: round(float(corr_matrix.loc[col, c]), 4) for c in cols} for col in cols}
        return corr_dict, high_corr_pairs, clusters

    def _compute_mutual_info(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[Dict[str, float], Dict[str, int]]:
        # Downsample X for fast MI if large
        if len(X) > 100000:
            idx = np.random.choice(len(X), 100000, replace=False)
            X_sub, y_sub = X.iloc[idx], y[idx]
        else:
            X_sub, y_sub = X, y

        scores = mutual_info_classif(X_sub, y_sub, random_state=42)
        mi_dict = {col: round(float(sc), 6) for col, sc in zip(X.columns, scores)}
        
        sorted_cols = sorted(mi_dict.keys(), key=lambda k: mi_dict[k], reverse=True)
        ranks = {col: rank + 1 for rank, col in enumerate(sorted_cols)}

        return mi_dict, ranks

    def _compute_anova_f(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[Dict[str, float], Dict[str, int]]:
        f_vals, _ = f_classif(X, y)
        f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
        f_dict = {col: round(float(sc), 4) for col, sc in zip(X.columns, f_vals)}

        sorted_cols = sorted(f_dict.keys(), key=lambda k: f_dict[k], reverse=True)
        ranks = {col: rank + 1 for rank, col in enumerate(sorted_cols)}

        return f_dict, ranks

    def _compute_rf_importance(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float]]:
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        
        if len(X) > 100000:
            idx = np.random.choice(len(X), 100000, replace=False)
            X_sub, y_sub = X.iloc[idx], y[idx]
        else:
            X_sub, y_sub = X, y

        rf.fit(X_sub, y_sub)
        importances = rf.feature_importances_

        rf_dict = {col: round(float(imp), 6) for col, imp in zip(X.columns, importances)}
        sorted_cols = sorted(rf_dict.keys(), key=lambda k: rf_dict[k], reverse=True)
        ranks = {col: rank + 1 for rank, col in enumerate(sorted_cols)}

        # Permutation importance on small validation sample
        perm_sub = X_sub.iloc[:5000]
        perm_y = y_sub[:5000]
        res = permutation_importance(rf, perm_sub, perm_y, n_repeats=3, random_state=42, n_jobs=-1)
        perm_dict = {col: round(float(imp), 6) for col, imp in zip(X.columns, res.importances_mean)}

        return rf_dict, ranks, perm_dict

    def _compute_xgb_importance(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[Dict[str, float], Dict[str, int]]:
        xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
        
        if len(X) > 100000:
            idx = np.random.choice(len(X), 100000, replace=False)
            X_sub, y_sub = X.iloc[idx], y[idx]
        else:
            X_sub, y_sub = X, y

        xgb.fit(X_sub, y_sub)
        importances = xgb.feature_importances_

        xgb_dict = {col: round(float(imp), 6) for col, imp in zip(X.columns, importances)}
        sorted_cols = sorted(xgb_dict.keys(), key=lambda k: xgb_dict[k], reverse=True)
        ranks = {col: rank + 1 for rank, col in enumerate(sorted_cols)}

        return xgb_dict, ranks

    def _build_evidence_table(
        self,
        cols: List[str],
        mi_scores: Dict[str, float], mi_ranks: Dict[str, int],
        f_scores: Dict[str, float], f_ranks: Dict[str, int],
        rf_imps: Dict[str, float], rf_ranks: Dict[str, int],
        xgb_imps: Dict[str, float], xgb_ranks: Dict[str, int],
        high_corr_pairs: List[Dict[str, Any]],
        dataset_name: str
    ) -> List[Dict[str, Any]]:
        corr_risk_cols = set([pair['feature_b'] for pair in high_corr_pairs])

        table = []
        for col in cols:
            mi_r = mi_ranks.get(col, 999)
            f_r = f_ranks.get(col, 999)
            rf_r = rf_ranks.get(col, 999)
            xgb_r = xgb_ranks.get(col, 999)

            avg_rank = round((mi_r + f_r + rf_r + xgb_r) / 4.0, 2)
            corr_risk = 'HIGH' if col in corr_risk_cols else 'LOW'

            table.append({
                'feature': col,
                'mi_score': mi_scores.get(col, 0.0),
                'mi_rank': mi_r,
                'f_score': f_scores.get(col, 0.0),
                'f_rank': f_r,
                'rf_importance': rf_imps.get(col, 0.0),
                'rf_rank': rf_r,
                'xgb_importance': xgb_imps.get(col, 0.0),
                'xgb_rank': xgb_r,
                'average_rank': avg_rank,
                'correlation_risk': corr_risk
            })

        table = sorted(table, key=lambda x: x['average_rank'])
        return table

    def _make_selection_decisions(
        self,
        evidence_table: List[Dict[str, Any]],
        dataset_name: str,
        pure_ids: List[str]
    ) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """
        Determines defensible final feature set based on multi-signal evidence table.
        Retains top informative features while eliminating extreme correlation redundancies.
        Assigns roles: TEMPORAL, RELATIONAL, BEHAVIORAL.
        """
        selected_features = []
        dropped_features = []

        # Target range: ~18 to 28 features depending on dataset capacity and evidence
        # NF-v2 has 43 features -> top ~20-25 selected
        # CSE-CIC has 80 features -> top ~25-30 selected

        max_select = 25 if 'nf' in dataset_name.lower() else 28
        corr_dropped = set()

        for item in evidence_table:
            feat = item['feature']
            risk = item['correlation_risk']
            avg_rank = item['average_rank']

            if len(selected_features) >= max_select:
                dropped_features.append({
                    'feature': feat,
                    'reason': f"Ranked below top {max_select} cutoff (Average rank {avg_rank})"
                })
                continue

            if risk == 'HIGH' and avg_rank > 15:
                corr_dropped.add(feat)
                dropped_features.append({
                    'feature': feat,
                    'reason': f"High correlation redundancy and average rank ({avg_rank}) > 15"
                })
                continue

            selected_features.append(feat)

        # Categorize roles for selected features
        temporal_feats = []
        relational_feats = []
        behavioral_feats = []

        for feat in selected_features:
            f_upper = feat.upper()
            if any(kw in f_upper for kw in ['DURATION', 'IAT', 'SECOND', 'THROUGHPUT', 'RATE', 'SPEED', 'FLOW_DURATION']):
                temporal_feats.append(feat)
            elif any(kw in f_upper for kw in ['PORT', 'PROTO', 'ICMP', 'DNS', 'TTL', 'IP']):
                relational_feats.append(feat)
            else:
                behavioral_feats.append(feat)

        role_mapping = {
            'temporal_features': temporal_feats,
            'relational_edge_features': relational_feats,
            'behavioral_features': behavioral_feats,
            'relational_node_identity_features': [c for c in pure_ids if 'IP' in c.upper() or 'PORT' in c.upper()]
        }

        return selected_features, dropped_features, role_mapping
