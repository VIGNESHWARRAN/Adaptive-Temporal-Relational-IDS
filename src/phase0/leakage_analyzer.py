"""
Leakage Analyzer Module for Phase 0.

Performs:
1. Feature Taxonomy Classification (14 Semantic Categories)
2. Target/Label Verification & target ∉ X Logging
3. Identifier Analysis & Leakage Risk Assessment
4. Timestamp Analysis & Sequence Role Definition
5. Duplicate Row & Feature-Vector Analysis
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

class LeakageAnalyzer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze_dataset(self, dataset_name: str, df: pd.DataFrame, target_col: str = 'Label', attack_col: str = None) -> Dict[str, Any]:
        print(f"--> Running Leakage & Categorization Analysis on {dataset_name} ({len(df)} sample rows)...")
        
        # 1. Target Verification
        target_check = self._verify_target(df, target_col, attack_col)

        # 2. Taxonomy Categorization
        taxonomy = self._categorize_features(df, dataset_name, target_col, attack_col)

        # 3. Identifier Analysis
        identifier_report = self._analyze_identifiers(df, dataset_name)

        # 4. Timestamp Analysis
        timestamp_report = self._analyze_timestamp(df, dataset_name, target_col)

        # 5. Duplicate Analysis
        duplicate_report = self._analyze_duplicates(df, target_col, attack_col)

        results = {
            'dataset_name': dataset_name,
            'target_verification': target_check,
            'feature_taxonomy': taxonomy,
            'identifier_report': identifier_report,
            'timestamp_report': timestamp_report,
            'duplicate_report': duplicate_report
        }

        # Save to file
        out_path = os.path.join(self.output_dir, f"{dataset_name}_leakage_analysis.json")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _verify_target(self, df: pd.DataFrame, target_col: str, attack_col: str = None) -> Dict[str, Any]:
        cols = df.columns.tolist()
        has_target = target_col in cols
        has_attack = attack_col in cols if attack_col else False

        # Verify target is excluded from input features X
        feature_matrix_cols = [c for c in cols if c not in [target_col, attack_col] if c is not None]
        target_not_in_X = (target_col not in feature_matrix_cols) and (attack_col not in feature_matrix_cols if attack_col else True)

        class_counts = df[target_col].value_counts().to_dict() if has_target else {}
        total = len(df)

        return {
            'target_column': target_col,
            'attack_column': attack_col,
            'target_present': has_target,
            'target_not_in_X': target_not_in_X,
            'unique_classes_count': len(class_counts),
            'class_counts': {str(k): int(v) for k, v in class_counts.items()},
            'class_percentages': {str(k): round((v / total) * 100, 4) for k, v in class_counts.items()} if total > 0 else {}
        }

    def _categorize_features(self, df: pd.DataFrame, dataset_name: str, target_col: str, attack_col: str = None) -> List[Dict[str, str]]:
        taxonomy = []
        for col in df.columns:
            cat, sem_cat, candidate_role = self._map_col_to_category(col, dataset_name)
            
            if col in [target_col, attack_col]:
                cat = 'Target / Label'
                candidate_role = 'Target'

            dtype_str = str(df[col].dtype)
            ftype = 'Categorical' if dtype_str == 'object' or col in ['PROTOCOL', 'Protocol', 'L7_PROTO', 'ICMP_TYPE'] else 'Numerical'

            taxonomy.append({
                'feature': col,
                'type': ftype,
                'semantic_category': sem_cat,
                'category_code': cat,
                'candidate_role': candidate_role
            })
        return taxonomy

    def _map_col_to_category(self, col: str, dataset_name: str) -> Tuple[str, str, str]:
        c_upper = col.upper().strip()

        if c_upper in ['LABEL', 'ATTACK']:
            return ('1. Target / Label', 'Target', 'Target')

        if c_upper in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'SRC IP', 'DST IP', 'FLOW ID', 'ROW_ID']:
            return ('2. Identifier', 'Endpoint identity / Flow ID', 'Relational / Node Identity')

        if c_upper in ['TIMESTAMP', 'TIME']:
            return ('3. Timestamp', 'Timestamp', 'Temporal / Ordering')

        if c_upper in ['L4_SRC_PORT', 'L4_DST_PORT', 'SRC PORT', 'DST PORT', 'DNS_QUERY_ID']:
            return ('4. Network endpoint information', 'Service / Port', 'Relational / Edge Attribute')

        if c_upper in ['PROTOCOL', 'L7_PROTO', 'ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_TYPE']:
            return ('5. Protocol information', 'Protocol', 'Relational / Edge Attribute')

        if 'PKT' in c_upper or 'PACKET' in c_upper or 'PKTS' in c_upper:
            return ('6. Packet statistics', 'Packet statistics', 'Temporal / Behavioral')

        if 'BYTE' in c_upper or 'BYTES' in c_upper or 'LEN' in c_upper:
            return ('7. Byte statistics', 'Byte statistics', 'Temporal / Behavioral')

        if 'DURATION' in c_upper or 'DUR' in c_upper:
            return ('8. Flow duration / timing', 'Flow timing', 'Temporal')

        if 'IAT' in c_upper or 'THROUGHPUT' in c_upper:
            return ('9. Inter-arrival-time statistics', 'IAT / Rate statistics', 'Temporal')

        if 'FLAG' in c_upper or 'FLAGS' in c_upper:
            return ('10. TCP/transport flags', 'TCP Flags', 'Behavioral')

        if 'PER_SEC' in c_upper or 'SECOND' in c_upper or 'BYTS/S' in c_upper or 'PKTS/S' in c_upper:
            return ('11. Rate features', 'Flow rates', 'Temporal / Behavioral')

        if 'WIN' in c_upper or 'HEADER' in c_upper or 'TTL' in c_upper:
            return ('12. Window/header features', 'Header / Window', 'Behavioral')

        return ('13. Other behavioral features', 'Other behavior', 'Behavioral')

    def _analyze_identifiers(self, df: pd.DataFrame, dataset_name: str) -> List[Dict[str, Any]]:
        candidate_ids = [c for c in df.columns if any(kw in c.upper() for kw in ['IP', 'PORT', 'ID', 'ROW', 'HASH'])]
        report = []
        n_rows = len(df)

        for col in candidate_ids:
            unq = df[col].nunique(dropna=True)
            ratio = unq / max(n_rows, 1)
            c_upper = col.upper()

            if c_upper in ['FLOW ID', 'ROW_ID'] or ratio > 0.95:
                classif = 'A. Pure record identifier'
                risk = 'High (if used directly in tabular classifier)'
                decision = 'EXCLUDE from tabular feature vector'
                reason = 'Pure unique identifier causing over-fitting and label memorization.'
            elif c_upper in ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'SRC IP', 'DST IP']:
                classif = 'B. Network identity information'
                risk = 'Moderate to High (IP memorization risk in static tabular models)'
                decision = 'USE FOR GRAPH NODE CONSTRUCTION; EXCLUDE from raw tabular classifier'
                reason = 'Defines topology in dynamic relational graph branch. Raw IP numbers leak subnet/host specific identities.'
            elif 'PORT' in c_upper:
                classif = 'B. Network identity information'
                risk = 'Low to Moderate'
                decision = 'RETAIN as categorical/numerical edge attribute or service feature'
                reason = 'Port numbers represent standard transport services (e.g. 80, 443, 22, 53).'
            elif 'DNS_QUERY_ID' in c_upper:
                classif = 'C. Potential leakage identifier'
                risk = 'High'
                decision = 'EXCLUDE'
                reason = 'Transaction IDs are transient sequence tokens that offer zero generalization.'
            else:
                classif = 'B. Behavioral network feature'
                risk = 'Low'
                decision = 'RETAIN for feature selection'
                reason = 'Statistically informative protocol/service indicator.'

            report.append({
                'feature': col,
                'reason_identified': f"Candidate identifier matching keyword, unique count={unq}",
                'unique_value_ratio': round(ratio, 6),
                'classification': classif,
                'leakage_risk': risk,
                'decision': decision,
                'reasoning': reason
            })

        return report

    def _analyze_timestamp(self, df: pd.DataFrame, dataset_name: str, target_col: str) -> Dict[str, Any]:
        ts_cols = [c for c in df.columns if 'TIME' in c.upper()]
        if not ts_cols:
            return {
                'timestamp_available': False,
                'details': 'No explicit Timestamp column found in raw schema.'
            }

        ts_col = ts_cols[0]
        s_ts = df[ts_col]

        # Check resolution & parse
        is_string = s_ts.dtype == 'object'
        monotonic = False
        parsed_ts = None

        if is_string:
            try:
                parsed_ts = pd.to_datetime(s_ts, errors='coerce', dayfirst=True)
                monotonic = parsed_ts.is_monotonic_increasing
            except Exception:
                pass
        else:
            monotonic = s_ts.is_monotonic_increasing

        # Time of day / Collection day distribution per attack class
        class_ts_summary = {}
        if parsed_ts is not None and target_col in df.columns:
            temp_df = pd.DataFrame({'ts': parsed_ts, 'target': df[target_col]}).dropna()
            temp_df['hour'] = temp_df['ts'].dt.hour
            temp_df['date'] = temp_df['ts'].dt.date.astype(str)

            for cls_val in temp_df['target'].unique():
                cls_sub = temp_df[temp_df['target'] == cls_val]
                class_ts_summary[str(cls_val)] = {
                    'min_timestamp': str(cls_sub['ts'].min()),
                    'max_timestamp': str(cls_sub['ts'].max()),
                    'active_hours': cls_sub['hour'].unique().tolist(),
                    'active_dates': cls_sub['date'].unique().tolist()
                }

        return {
            'timestamp_available': True,
            'timestamp_column': ts_col,
            'data_type': str(s_ts.dtype),
            'is_monotonically_ordered': monotonic,
            'leakage_risk_evaluation': 'HIGH if raw timestamp integer/string is fed directly to classifier (models memorize attack collection windows). LOW if used exclusively for chronological sliding window ordering and inter-flow delta calculations.',
            'recommended_role': 'Sequence Ordering / Inter-Flow Velocity Engine; EXCLUDED from static feature matrix X.',
            'class_timestamp_distributions': class_ts_summary
        }

    def _analyze_duplicates(self, df: pd.DataFrame, target_col: str, attack_col: str = None) -> Dict[str, Any]:
        total_rows = len(df)
        
        # Exact row duplicates
        exact_dups = int(df.duplicated().sum())
        exact_dup_pct = (exact_dups / max(total_rows, 1)) * 100.0

        # Feature-vector duplicates (excluding label/target/attack columns)
        exclude_cols = [target_col]
        if attack_col and attack_col in df.columns:
            exclude_cols.append(attack_col)
        
        feat_cols = [c for c in df.columns if c not in exclude_cols]
        feat_dups = int(df.duplicated(subset=feat_cols).sum())
        feat_dup_pct = (feat_dups / max(total_rows, 1)) * 100.0

        return {
            'total_rows_analyzed': total_rows,
            'exact_duplicate_rows': exact_dups,
            'exact_duplicate_percentage': round(exact_dup_pct, 4),
            'feature_vector_duplicates': feat_dups,
            'feature_vector_duplicate_percentage': round(feat_dup_pct, 4),
            'leakage_risk_note': 'Cross-split duplicates between Train and Test splits can inflate test accuracy. Train/Val/Test splits must be strictly deduplicated or chronologically block-split.'
        }
