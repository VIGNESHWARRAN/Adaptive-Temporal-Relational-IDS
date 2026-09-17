"""
Dataset Profiler Module for Phase 0.

Profiles datasets at the dataset-level and feature-level:
- Record count, column count, target column, class distribution, imbalance ratio.
- Feature-level stats: dtype, unique count, missing count/%, inf count/%, zero %, min, max, mean, median, std, top categories.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

class DatasetProfiler:
    def __init__(self, data_paths: Dict[str, Any], output_dir: str):
        """
        data_paths: Dictionary containing raw paths for:
          - 'cse_cic_ids2018': dir path containing the 10 CSVs
          - 'nf_cse_cic_ids2018_v2': csv file path
          - 'nf_unsw_nb15_v2': csv file path
        output_dir: directory to store profiles and reports
        """
        self.data_paths = data_paths
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def profile_all(self, sample_limit: int = 1000000) -> Dict[str, Any]:
        """
        Runs profiling on all three datasets.
        sample_limit: limit per dataset for fine-grained feature stats if full stream is large.
        """
        all_profiles = {}
        all_profiles['cse_cic_ids2018'] = self.profile_cse_cic_ids2018(sample_limit=sample_limit)
        all_profiles['nf_cse_cic_ids2018_v2'] = self.profile_nf_v2('nf_cse_cic_ids2018_v2', self.data_paths['nf_cse_cic_ids2018_v2'], sample_limit=sample_limit)
        all_profiles['nf_unsw_nb15_v2'] = self.profile_nf_v2('nf_unsw_nb15_v2', self.data_paths['nf_unsw_nb15_v2'], sample_limit=sample_limit)

        self._export_combined_summary(all_profiles)
        return all_profiles

    def profile_nf_v2(self, dataset_name: str, file_path: str, sample_limit: int = 1000000) -> Dict[str, Any]:
        print(f"--> Profiling {dataset_name} from {file_path}...")
        
        # 1. First pass: count total rows and class distribution using streaming chunks
        total_records = 0
        class_counts = {}
        target_col = 'Label'
        attack_col = 'Attack'
        
        chunksize = 100000
        first_chunk = None

        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            if first_chunk is None:
                first_chunk = chunk
            total_records += len(chunk)
            
            # Count binary target
            if target_col in chunk.columns:
                vc = chunk[target_col].value_counts().to_dict()
                for k, v in vc.items():
                    class_counts[str(k)] = class_counts.get(str(k), 0) + int(v)

        cols = first_chunk.columns.tolist()
        num_cols = len(cols)
        
        # Imbalance ratio
        counts_list = list(class_counts.values())
        max_c = max(counts_list) if counts_list else 1
        min_c = min(counts_list) if counts_list else 1
        imbalance_ratio = max_c / max(min_c, 1)

        # 2. Second pass: Sample up to sample_limit rows for feature-level profile
        if total_records > sample_limit:
            frac = sample_limit / total_records
            sampled_dfs = []
            for chunk in pd.read_csv(file_path, chunksize=chunksize):
                sample_chunk = chunk.sample(frac=frac, random_state=42)
                sampled_dfs.append(sample_chunk)
            df_sample = pd.concat(sampled_dfs, ignore_index=True)
        else:
            df_sample = pd.read_csv(file_path)

        # 3. Compute feature statistics
        feature_stats = self._compute_feature_stats(df_sample)

        profile = {
            'dataset_name': dataset_name,
            'file_format': 'CSV',
            'file_size_bytes': os.path.getsize(file_path),
            'total_records': total_records,
            'profile_sample_records': len(df_sample),
            'num_columns': num_cols,
            'candidate_features': num_cols - 2 if 'Attack' in cols else num_cols - 1,
            'target_column': target_col,
            'num_classes': len(class_counts),
            'class_distribution': class_counts,
            'class_percentages': {k: (v / total_records) * 100 for k, v in class_counts.items()},
            'imbalance_ratio': round(imbalance_ratio, 4),
            'feature_stats': feature_stats
        }

        # Save profile JSON
        out_path = os.path.join(self.output_dir, f"{dataset_name}_profile.json")
        with open(out_path, 'w') as f:
            json.dump(profile, f, indent=2)

        return profile

    def profile_cse_cic_ids2018(self, sample_limit: int = 1000000) -> Dict[str, Any]:
        dir_path = self.data_paths['cse_cic_ids2018']
        csv_files = sorted(glob.glob(os.path.join(dir_path, '*.csv')))
        print(f"--> Profiling CSE-CIC-IDS2018 across {len(csv_files)} CSV files...")

        total_records = 0
        total_size = sum(os.path.getsize(f) for f in csv_files)
        class_counts = {}
        
        # Determine schema & total records
        chunksize = 100000
        sampled_dfs = []
        target_col = 'Label'
        
        # Calculate file sizes and total row estimate
        for f in csv_files:
            for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False):
                # Clean column headers
                chunk.columns = [c.strip() for c in chunk.columns]
                # Fix header rows inside data if present
                if target_col in chunk.columns:
                    mask = chunk[target_col] != target_col
                    chunk = chunk[mask]
                
                total_records += len(chunk)
                if target_col in chunk.columns:
                    vc = chunk[target_col].value_counts().to_dict()
                    for k, v in vc.items():
                        class_counts[str(k)] = class_counts.get(str(k), 0) + int(v)

        # Sample uniform fraction across files
        frac = min(1.0, sample_limit / max(total_records, 1))
        for f in csv_files:
            for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False):
                chunk.columns = [c.strip() for c in chunk.columns]
                if target_col in chunk.columns:
                    chunk = chunk[chunk[target_col] != target_col]
                s_chunk = chunk.sample(frac=frac, random_state=42) if frac < 1.0 else chunk
                sampled_dfs.append(s_chunk)

        df_sample = pd.concat(sampled_dfs, ignore_index=True)
        # Drop extra columns that appear only in 02-20-2018.csv if we want common candidate features or profile all 84
        all_cols = df_sample.columns.tolist()

        counts_list = list(class_counts.values())
        max_c = max(counts_list) if counts_list else 1
        min_c = min(counts_list) if counts_list else 1
        imbalance_ratio = max_c / max(min_c, 1)

        feature_stats = self._compute_feature_stats(df_sample)

        profile = {
            'dataset_name': 'cse_cic_ids2018',
            'file_format': 'CSV (10 files)',
            'file_size_bytes': total_size,
            'total_records': total_records,
            'profile_sample_records': len(df_sample),
            'num_columns': len(all_cols),
            'candidate_features': len(all_cols) - 1,
            'target_column': target_col,
            'num_classes': len(class_counts),
            'class_distribution': class_counts,
            'class_percentages': {k: (v / total_records) * 100 for k, v in class_counts.items()},
            'imbalance_ratio': round(imbalance_ratio, 4),
            'feature_stats': feature_stats
        }

        out_path = os.path.join(self.output_dir, "cse_cic_ids2018_profile.json")
        with open(out_path, 'w') as f:
            json.dump(profile, f, indent=2)

        return profile

    def _compute_feature_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        stats_list = []
        n_rows = len(df)

        for col in df.columns:
            series = df[col]
            if col.upper().strip() not in ['LABEL', 'ATTACK', 'TIMESTAMP', 'TIME', 'FLOW ID', 'SRC IP', 'DST IP', 'SRC_IP', 'DST_IP', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR']:
                if series.dtype == 'object':
                    num_converted = pd.to_numeric(series, errors='coerce')
                    if num_converted.notnull().sum() > 0.5 * len(series):
                        series = num_converted
            dtype_str = str(series.dtype)

            # Missing count & %
            missing_count = int(series.isnull().sum())
            missing_pct = (missing_count / n_rows) * 100.0

            # Inf count
            if np.issubdtype(series.dtype, np.number):
                inf_count = int(np.isinf(series).sum())
            else:
                inf_count = 0
            inf_pct = (inf_count / n_rows) * 100.0

            # Zero count & %
            if np.issubdtype(series.dtype, np.number):
                clean_s = series.replace([np.inf, -np.inf], np.nan).dropna().clip(-1e12, 1e12)
                zero_count = int((clean_s == 0).sum())
                zero_pct = (zero_count / max(len(clean_s), 1)) * 100.0
                unique_val_count = int(series.nunique(dropna=True))

                min_val = float(clean_s.min()) if len(clean_s) > 0 else None
                max_val = float(clean_s.max()) if len(clean_s) > 0 else None
                mean_val = float(clean_s.mean()) if len(clean_s) > 0 else None
                median_val = float(clean_s.median()) if len(clean_s) > 0 else None
                std_val = float(clean_s.std()) if len(clean_s) > 0 else None

                top_cats = None
            else:
                clean_s = series.dropna()
                zero_pct = 0.0
                unique_val_count = int(series.nunique(dropna=True))
                min_val, max_val, mean_val, median_val, std_val = None, None, None, None, None
                
                # Categorical top categories
                top_vc = series.value_counts(dropna=True).head(5).to_dict()
                top_cats = {str(k): int(v) for k, v in top_vc.items()}

            stats_list.append({
                'feature_name': col,
                'dtype': dtype_str,
                'unique_values': unique_val_count,
                'missing_count': missing_count,
                'missing_percentage': round(missing_pct, 4),
                'infinite_count': inf_count,
                'infinite_percentage': round(inf_pct, 4),
                'zero_percentage': round(zero_pct, 4),
                'min': min_val,
                'max': max_val,
                'mean': round(mean_val, 4) if mean_val is not None else None,
                'median': round(median_val, 4) if median_val is not None else None,
                'std': round(std_val, 4) if std_val is not None else None,
                'top_categories': top_cats
            })

        return stats_list

    def _export_combined_summary(self, profiles: Dict[str, Any]):
        """
        Exports dataset_profile.csv containing all column-level statistics for all 3 datasets.
        """
        rows = []
        for ds_name, prof in profiles.items():
            for fstat in prof['feature_stats']:
                row = {'dataset_name': ds_name}
                row.update(fstat)
                rows.append(row)

        df_prof = pd.DataFrame(rows)
        csv_path = os.path.join(self.output_dir, 'dataset_profile.csv')
        df_prof.to_csv(csv_path, index=False)
        print(f"--> Exported dataset_profile.csv with {len(df_prof)} rows to {csv_path}")
