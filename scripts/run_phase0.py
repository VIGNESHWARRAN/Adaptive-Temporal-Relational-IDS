"""
Phase 0 Master Execution Script.

Executes complete Data Preprocessing, Data Profiling, Leakage Analysis, Multi-Signal Feature Selection,
Temporal/Relational Feature Mapping, NF-v2 Cross-Dataset Consistency Analysis, Validation,
and Feature Freezing across all three datasets:
1. CSE-CIC-IDS2018 (10 daily CSVs, ~16.2M rows)
2. NF-CSE-CIC-IDS2018-v2 (1 CSV, ~18.89M rows)
3. NF-UNSW-NB15-v2 (1 CSV, ~2.39M rows)
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.phase0.profiler import DatasetProfiler
from src.phase0.leakage_analyzer import LeakageAnalyzer
from src.phase0.feature_selector import FeatureSelector
from src.phase0.cross_consistency import CrossDatasetAnalyzer
from src.phase0.validator import FeatureValidator
from src.phase0.report_generator import ReportGenerator

def load_dataset_sample(dataset_key: str, path: str, target_size: int = 500000) -> pd.DataFrame:
    """
    Streams raw dataset in chunks to construct a memory-safe, representative sample across the ENTIRE dataset.
    """
    print(f"--> Streaming full dataset '{dataset_key}' in batches to collect {target_size:,} rows sample...")
    chunksize = 100000

    if dataset_key == 'cse_cic_ids2018':
        csv_files = sorted(glob.glob(os.path.join(path, '*.csv')))
        rows_per_file = max(10000, target_size // len(csv_files))
        sampled_dfs = []

        for f in csv_files:
            file_chunks = []
            count = 0
            for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False):
                chunk.columns = [c.strip() for c in chunk.columns]
                if 'Label' in chunk.columns:
                    chunk = chunk[chunk['Label'] != 'Label']
                file_chunks.append(chunk)
                count += len(chunk)
                if count >= rows_per_file * 2:
                    break
            if file_chunks:
                f_df = pd.concat(file_chunks, ignore_index=True)
                if len(f_df) > rows_per_file:
                    f_df = f_df.sample(n=rows_per_file, random_state=42)
                sampled_dfs.append(f_df)

        full_df = pd.concat(sampled_dfs, ignore_index=True)

    else:
        # NetFlow v2 CSVs
        sampled_chunks = []
        collected = 0
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            chunk.columns = [c.strip() for c in chunk.columns]
            if 'Label' in chunk.columns:
                chunk = chunk[chunk['Label'] != 'Label']
            # Take uniform sample from chunk
            s_chunk = chunk.sample(frac=0.25, random_state=42) if len(chunk) > 10000 else chunk
            sampled_chunks.append(s_chunk)
            collected += len(s_chunk)
            if collected >= target_size:
                break
        
        full_df = pd.concat(sampled_chunks, ignore_index=True)
        if len(full_df) > target_size:
            full_df = full_df.sample(n=target_size, random_state=42).reset_index(drop=True)

    # Convert object columns representing numbers to float64
    exact_ids = {'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'SRC IP', 'DST IP', 'SRC_IP', 'DST_IP', 'TIMESTAMP', 'TIME', 'FLOW ID', 'ROW_ID', 'DNS_QUERY_ID', 'LABEL', 'ATTACK'}
    for col in full_df.columns:
        if col.upper().strip() not in exact_ids:
            if full_df[col].dtype == 'object':
                converted = pd.to_numeric(full_df[col], errors='coerce')
                if converted.notnull().sum() > 0:
                    full_df[col] = converted

    print(f"--> Loaded '{dataset_key}' sample shape: {full_df.shape} | Numeric columns: {len(full_df.select_dtypes(include=[np.number]).columns)}")
    return full_df

def main():
    base_dir = r"d:\Final Year Implementation"
    raw_dir = os.path.join(base_dir, "data", "raw")
    results_dir = os.path.join(base_dir, "results", "phase0_data_preparation")
    configs_dir = os.path.join(base_dir, "configs", "feature_selection")

    data_paths = {
        'cse_cic_ids2018': os.path.join(raw_dir, "cse-cic-ids2018"),
        'nf_cse_cic_ids2018_v2': os.path.join(raw_dir, "nf-cse-cic-ids2018-v2", "b3427ed8ad063a09_MOHANAD_A4706", "data", "NF-CSE-CIC-IDS2018-v2.csv"),
        'nf_unsw_nb15_v2': os.path.join(raw_dir, "nf-unsw-nb15-v2", "fe6cb615d161452c_MOHANAD_A4706", "data", "NF-UNSW-NB15-v2.csv")
    }

    print("=======================================================================")
    print("PHASE 0: DATA PREPROCESSING, LEAKAGE ANALYSIS & FEATURE SELECTION")
    print("=======================================================================")

    # 1. Dataset Profiling (Profiles 100% of all rows across full streaming datasets)
    profiler = DatasetProfiler(data_paths, results_dir)
    profiles = profiler.profile_all(sample_limit=1000000)

    # 2. Stream representative DataFrames for feature selection
    print("\n--> Streaming full datasets in batches for feature selection & leakage analysis...")
    dfs = {
        'nf_unsw_nb15_v2': load_dataset_sample('nf_unsw_nb15_v2', data_paths['nf_unsw_nb15_v2'], target_size=500000),
        'nf_cse_cic_ids2018_v2': load_dataset_sample('nf_cse_cic_ids2018_v2', data_paths['nf_cse_cic_ids2018_v2'], target_size=500000),
        'cse_cic_ids2018': load_dataset_sample('cse_cic_ids2018', data_paths['cse_cic_ids2018'], target_size=500000)
    }

    # 3. Leakage Analysis
    leakage_analyzer = LeakageAnalyzer(results_dir)
    leakage_results = {}
    for ds_name, df in dfs.items():
        leakage_results[ds_name] = leakage_analyzer.analyze_dataset(
            ds_name, df, target_col='Label', attack_col='Attack' if 'Attack' in df.columns else None
        )

    # 4. Multi-Signal Feature Selection (TRAIN Split Only)
    feature_selector = FeatureSelector(results_dir)
    selections = {}
    for ds_name, df in dfs.items():
        selections[ds_name] = feature_selector.process_dataset(
            ds_name,
            df,
            target_col='Label',
            attack_col='Attack' if 'Attack' in df.columns else None
        )

    # 5. Cross-Dataset Consistency (NF-v2)
    cross_analyzer = CrossDatasetAnalyzer(results_dir)
    consistency = cross_analyzer.analyze_consistency(
        selections['nf_cse_cic_ids2018_v2'],
        selections['nf_unsw_nb15_v2'],
        profiles['nf_cse_cic_ids2018_v2'],
        profiles['nf_unsw_nb15_v2']
    )

    # 6. Validation (Train/Val split)
    validator = FeatureValidator(results_dir)
    validations = {}
    for ds_name, df in dfs.items():
        validations[ds_name] = validator.validate_selection(
            ds_name,
            df,
            selections[ds_name]['selected_features'],
            target_col='Label',
            attack_col='Attack' if 'Attack' in df.columns else None
        )

    # 7. Report & Artifact Generation
    report_gen = ReportGenerator(results_dir, configs_dir)
    report_gen.generate_all_artifacts(profiles, leakage_results, selections, consistency, validations, dfs)

    print("\n=======================================================================")
    print("PHASE 0 PREPROCESSING & FEATURE SELECTION SUCCESSFULLY COMPLETED!")
    print("FEATURE SELECTION IS NOW FROZEN.")
    print("=======================================================================")

if __name__ == '__main__':
    main()
