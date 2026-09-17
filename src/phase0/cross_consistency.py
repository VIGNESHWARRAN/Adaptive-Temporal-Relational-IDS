"""
NF-v2 Cross-Dataset Consistency Analyzer for Phase 0 (Step 17).

Compares feature behavior and importance ranks between:
- NF-CSE-CIC-IDS2018-v2
- NF-UNSW-NB15-v2
across their shared 43 standardized NetFlow-v2 extended features.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class CrossDatasetAnalyzer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze_consistency(
        self,
        nf_cse_selection: Dict[str, Any],
        nf_unsw_selection: Dict[str, Any],
        nf_cse_profile: Dict[str, Any],
        nf_unsw_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        print("\n--> Running NF-v2 Cross-Dataset Feature Consistency Analysis...")

        # Extract evidence tables
        cse_table = {item['feature']: item for item in nf_cse_selection['evidence_table']}
        unsw_table = {item['feature']: item for item in nf_unsw_selection['evidence_table']}

        cse_prof_stats = {item['feature_name']: item for item in nf_cse_profile['feature_stats']}
        unsw_prof_stats = {item['feature_name']: item for item in nf_unsw_profile['feature_stats']}

        shared_features = sorted(list(set(cse_table.keys()).intersection(set(unsw_table.keys()))))
        print(f"Found {len(shared_features)} shared features between NF-CSE and NF-UNSW.")

        comparison_list = []
        consistently_useful = []
        dataset_specific = []
        unstable_features = []

        for feat in shared_features:
            cse_item = cse_table[feat]
            unsw_item = unsw_table[feat]

            cse_mi_r = cse_item['mi_rank']
            unsw_mi_r = unsw_item['mi_rank']

            cse_rf_r = cse_item['rf_rank']
            unsw_rf_r = unsw_item['rf_rank']

            cse_avg_r = cse_item['average_rank']
            unsw_avg_r = unsw_item['average_rank']

            rank_diff = abs(cse_avg_r - unsw_avg_r)

            # Classify consistency
            if cse_avg_r <= 20 and unsw_avg_r <= 20:
                classification = 'Consistently Useful'
                consistently_useful.append(feat)
            elif (cse_avg_r <= 15 and unsw_avg_r > 25) or (unsw_avg_r <= 15 and cse_avg_r > 25):
                classification = 'Dataset-Specific'
                dataset_specific.append(feat)
            else:
                classification = 'Potentially Unstable' if rank_diff > 12 else 'Moderately Consistent'
                if rank_diff > 12:
                    unstable_features.append(feat)

            cse_stat = cse_prof_stats.get(feat, {})
            unsw_stat = unsw_prof_stats.get(feat, {})

            comparison_list.append({
                'feature': feat,
                'available_in_both': True,
                'nf_cse_mi_rank': cse_mi_r,
                'nf_unsw_mi_rank': unsw_mi_r,
                'nf_cse_rf_rank': cse_rf_r,
                'nf_unsw_rf_rank': unsw_rf_r,
                'nf_cse_average_rank': cse_avg_r,
                'nf_unsw_average_rank': unsw_avg_r,
                'rank_difference': round(rank_diff, 2),
                'consistency_classification': classification,
                'nf_cse_mean': cse_stat.get('mean'),
                'nf_unsw_mean': unsw_stat.get('mean')
            })

        comparison_list = sorted(comparison_list, key=lambda x: (x['nf_cse_average_rank'] + x['nf_unsw_average_rank']) / 2.0)

        results = {
            'total_shared_features': len(shared_features),
            'consistently_useful_count': len(consistently_useful),
            'consistently_useful_features': consistently_useful,
            'dataset_specific_count': len(dataset_specific),
            'dataset_specific_features': dataset_specific,
            'unstable_count': len(unstable_features),
            'unstable_features': unstable_features,
            'feature_comparisons': comparison_list
        }

        # Save JSON
        out_path = os.path.join(self.output_dir, "nf_v2_cross_dataset_consistency.json")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"--> Cross-dataset analysis complete: {len(consistently_useful)} consistently useful, {len(dataset_specific)} dataset-specific, {len(unstable_features)} unstable.")
        return results
