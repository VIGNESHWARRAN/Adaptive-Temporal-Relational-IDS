"""
Metadata Reporter Module.
Generates structured Markdown and JSON reports adhering to the research dataset metadata template.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class MetadataReporter:
    """Formats and exports dataset metadata into Markdown and JSON artifacts."""

    def __init__(self, output_dir: str = "metadata_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(self, profile: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """Export raw profile dictionary to JSON file."""
        if filename is None:
            safe_name = profile.get("dataset_name", "dataset").replace(" ", "_").lower()
            filename = f"{safe_name}_metadata.json"

        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
        print(f"[SUCCESS] JSON Metadata saved: {out_path}")
        return out_path

    def export_markdown(self, profile: Dict[str, Any], registry_meta: Optional[Dict[str, Any]] = None, filename: Optional[str] = None) -> Path:
        """
        Generate a Markdown metadata report conforming to the research metadata template.
        """
        if filename is None:
            safe_name = profile.get("dataset_name", "dataset").replace(" ", "_").lower()
            filename = f"{safe_name}_metadata.md"

        out_path = self.output_dir / filename
        reg = registry_meta or {}
        tr_eval = profile.get("temporal_relational_evaluation", {})
        rel_fields = tr_eval.get("relational_fields", {})
        temp_fields = tr_eval.get("temporal_fields", {})
        vol_fields = tr_eval.get("volume_fields", {})

        md = []
        md.append(f"# Dataset Metadata: {profile.get('dataset_name', 'Unknown')}")
        md.append(f"\n*Generated automatically by Adaptive Temporal-Relational IDS Engine on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        md.append("---\n")

        # A. Basic Information
        md.append("## A. Basic Information")
        md.append(f"- **dataset_name:** {reg.get('name', profile.get('dataset_name'))}")
        md.append(f"- **dataset_version:** {reg.get('version', 'N/A')}")
        md.append(f"- **publication_year:** {reg.get('publication_year', 'N/A')}")
        md.append(f"- **dataset_release_year:** {reg.get('publication_year', 'N/A')}")
        md.append(f"- **authors/creators:** {reg.get('creators', 'N/A')}")
        md.append(f"- **official_source:** {reg.get('direct_url', 'N/A')}")
        md.append(f"- **kaggle_source:** {reg.get('kaggle_slug', 'N/A')}")
        md.append(f"- **role_tier:** Tier {reg.get('tier', 'Unassigned')} - {reg.get('role', 'N/A')}")
        md.append(f"- **description:** {reg.get('description', 'N/A')}\n")

        # B. Dataset Type & Lineage
        md.append("## B. Dataset Type & Lineage")
        md.append(f"- **dataset_type:** Flow / NetFlow (Format: `{profile.get('file_name', '').split('.')[-1]}`)")
        md.append(f"- **schema_type:** {reg.get('schema_type', 'Custom')}")
        md.append(f"- **underlying_traffic_source:** {reg.get('role', 'Independent PCAP capture')}\n")

        # C. Dataset Size & Storage Metrics
        size_info = profile.get("file_size", {})
        md.append("## C. Dataset Size & Physical Metrics")
        md.append(f"- **file_name:** `{profile.get('file_name')}`")
        md.append(f"- **file_size_mb:** {size_info.get('size_mb', 0)} MB ({size_info.get('size_gb', 0)} GB)")
        md.append(f"- **total_records:** {profile.get('total_records_estimate', 0):,}")
        md.append(f"- **total_features:** {profile.get('total_features', 0)}\n")

        # D. Critical Features for Temporal-Relational Research
        md.append("## D. Critical Features for Temporal-Relational Research")
        md.append(f"- **Temporal-Relational Compatibility Score:** **{tr_eval.get('score', 0)} / 100**")
        md.append(f"- **Graph Construction Ready ($G_t = (V_t, E_t)$):** `{'YES' if tr_eval.get('graph_ready') else 'NO'}`")
        md.append(f"- **Temporal Ordering Ready:** `{'YES' if tr_eval.get('temporal_order_ready') else 'NO'}`")
        md.append(f"- **NetFlow 5-Tuple Completeness:** `{'YES' if tr_eval.get('netflow_standard_ready') else 'NO'}`\n")
        
        md.append("### Endpoint Identifiers (Graph Nodes & Edges)")
        md.append(f"- **Source IP:** `{rel_fields.get('source_ip') or 'Not Found'}`")
        md.append(f"- **Destination IP:** `{rel_fields.get('destination_ip') or 'Not Found'}`")
        md.append(f"- **Source Port:** `{rel_fields.get('source_port') or 'Not Found'}`")
        md.append(f"- **Destination Port:** `{rel_fields.get('destination_port') or 'Not Found'}`")
        md.append(f"- **Protocol:** `{rel_fields.get('protocol') or 'Not Found'}`\n")

        # E. Temporal Features
        md.append("## E. Temporal Dynamics & Flow Durations")
        md.append(f"- **Timestamp Feature:** `{temp_fields.get('timestamp') or 'Not Found'}`")
        if profile.get("temporal_summary"):
            ts_sum = profile["temporal_summary"]
            md.append(f"  - *Sample Value:* `{ts_sum.get('sample_value')}`")
            md.append(f"  - *Is Numeric / Epoch:* `{ts_sum.get('is_numeric')}`")
        md.append(f"- **Flow Duration Feature:** `{temp_fields.get('duration') or 'Not Found'}`")
        md.append(f"- **Inter-Arrival Time (IAT) Features:** `{', '.join(temp_fields.get('iat_features', [])) or 'None'}`")
        md.append(f"- **Active / Idle Time Features:** `{', '.join(temp_fields.get('active_idle_features', [])) or 'None'}`\n")

        # F. Volume and Rate Statistics
        md.append("## F. Flow Volume & Rate Indicators")
        md.append(f"- **Packet Count Columns:** `{', '.join(vol_fields.get('packet_features', [])) or 'None'}`")
        md.append(f"- **Byte Count Columns:** `{', '.join(vol_fields.get('byte_features', [])) or 'None'}`")
        md.append(f"- **Flow Rate Columns:** `{', '.join(vol_fields.get('rate_features', [])) or 'None'}`\n")

        # G. Label & Class Distribution
        md.append("## G. Label & Ground Truth Distribution")
        lbl_dist = profile.get("labels_distribution", {})
        if not lbl_dist:
            md.append("- *No standard label columns identified in initial scan.*\n")
        else:
            for lbl_col, counts in lbl_dist.items():
                md.append(f"### Column: `{lbl_col}`")
                md.append("| Class / Label | Sample Count |")
                md.append("| :--- | :--- |")
                for cls_val, cnt in counts.items():
                    md.append(f"| `{cls_val}` | {cnt:,} |")
                md.append("")

        # H. Complete Column Schema Table
        md.append("## H. Complete Column Schema")
        md.append("| # | Column Name | Data Type | Null % | Sample Values |")
        md.append("| :--- | :--- | :--- | :--- | :--- |")
        for i, col in enumerate(profile.get("column_profiles", []), 1):
            s_val = ", ".join([str(v) for v in col.get("sample_values", [])])
            if len(s_val) > 40:
                s_val = s_val[:37] + "..."
            md.append(f"| {i} | `{col['column_name']}` | `{col['data_type']}` | {col['null_percent']}% | `{s_val}` |")

        content = "\n".join(md)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[SUCCESS] Markdown Report saved: {out_path}")
        return out_path
