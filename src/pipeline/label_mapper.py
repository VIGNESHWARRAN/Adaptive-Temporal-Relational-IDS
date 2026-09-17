"""
Harmonized 3-Tier Label Mapper.
Maps dataset-specific attack strings into Level 1 (Binary), Level 2 (8 Canonical Families), and Level 3.
Strictly requires explicit mapping rules without silent default fallbacks.
"""

from typing import Dict, Tuple, Any
import pandas as pd
import yaml


class HarmonizedLabelMapper:
    def __init__(self, config_path: str = "configs/label_taxonomy.yaml"):
        with open(config_path, "r") as f:
            self.taxonomy = yaml.safe_load(f).get("canonical_families", {})
        
        self.raw_to_family: Dict[str, str] = {}
        self.raw_to_binary: Dict[str, int] = {}

        for family, info in self.taxonomy.items():
            level_1 = info["level_1"]
            for match_str in info["raw_matches"]:
                self.raw_to_family[match_str] = family
                self.raw_to_binary[match_str] = level_1

    def map_label(self, raw_label: Any) -> Tuple[int, str, str]:
        """
        Maps a single raw label string to (label_binary, label_attack_family, label_specific_attack).
        Raises KeyError if an unmapped attack label string is encountered.
        """
        if pd.isna(raw_label):
            return 0, "BENIGN", "Benign"

        raw_str = str(raw_label).strip().lower()

        if str(raw_label) == "0" or raw_str == "0":
            return 0, "BENIGN", "Benign"

        for match_key, family in self.raw_to_family.items():
            if match_key in raw_str:
                return self.raw_to_binary[match_key], family, str(raw_label)

        # Explicit exception - no silent fallback to dummy label
        raise KeyError(
            f"Unmapped attack label string encountered: '{raw_label}'. "
            f"Add an explicit mapping rule for this label to configs/label_taxonomy.yaml."
        )

    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies 3-tier label mapping to a DataFrame containing 'label_binary' or 'label_specific_attack'.
        """
        df = df.copy()
        
        target_col = "label_specific_attack" if "label_specific_attack" in df.columns else "label_binary"
        
        if target_col in df.columns:
            mapped_tuples = df[target_col].apply(self.map_label)
            df["label_binary"] = [t[0] for t in mapped_tuples]
            df["label_attack_family"] = [t[1] for t in mapped_tuples]
            df["label_specific_attack"] = [t[2] for t in mapped_tuples]

        return df
