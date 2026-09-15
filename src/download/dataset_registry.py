"""
Dataset Registry Module.
Loads dataset definitions from YAML/JSON configuration and manages dataset metadata targets.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class DatasetRegistry:
    """Manages candidate IDS datasets, schemas, download endpoints, and roles."""

    def __init__(self, config_path: Optional[str] = None):
        pkg_root = Path(__file__).resolve().parent.parent.parent
        
        if config_path is None:
            # Check JSON first (no 3rd party deps required) then YAML
            json_path = pkg_root / "configs" / "datasets_config.json"
            yaml_path = pkg_root / "configs" / "datasets_config.yaml"
            if json_path.exists():
                self.config_path = json_path
            else:
                self.config_path = yaml_path
        else:
            self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Dataset config not found at: {self.config_path}")
            
        if self.config_path.suffix.lower() == ".json":
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._raw_config = json.load(f)
        else:
            try:
                import yaml
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._raw_config = yaml.safe_load(f)
            except ImportError:
                # If PyYAML is missing, attempt to load JSON version
                json_fallback = self.config_path.with_suffix(".json")
                if json_fallback.exists():
                    with open(json_fallback, "r", encoding="utf-8") as f:
                        self._raw_config = json.load(f)
                else:
                    raise ImportError("PyYAML is not installed and no fallback JSON config found. Please install pyyaml or provide JSON.")

        self.datasets: Dict[str, Dict[str, Any]] = self._raw_config.get("datasets", {})
        self.storage: Dict[str, Any] = self._raw_config.get("storage", {})

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Retrieve configuration dictionary for a given dataset ID (case-insensitive)."""
        key = dataset_id.lower().strip()
        if key not in self.datasets:
            available = ", ".join(self.datasets.keys())
            raise KeyError(f"Dataset '{dataset_id}' not found. Available datasets: {available}")
        return self.datasets[key]

    def list_datasets(self, tier: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all datasets, optionally filtered by tier (1, 2, or 3)."""
        result = []
        for ds_id, meta in self.datasets.items():
            if tier is None or meta.get("tier") == tier:
                entry = meta.copy()
                entry["id"] = ds_id
                result.append(entry)
        return result

    def get_all_ids(self) -> List[str]:
        """Get list of all available dataset IDs."""
        return list(self.datasets.keys())

    def get_tier1_ids(self) -> List[str]:
        """Get IDs of Tier 1 core primary datasets."""
        return [ds_id for ds_id, meta in self.datasets.items() if meta.get("tier") == 1]

    def get_tier2_ids(self) -> List[str]:
        """Get IDs of Tier 2 cross-domain and quality-sensitivity datasets."""
        return [ds_id for ds_id, meta in self.datasets.items() if meta.get("tier") == 2]
