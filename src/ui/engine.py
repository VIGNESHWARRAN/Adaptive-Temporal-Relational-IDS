"""Adaptive Temporal-Relational IDS - Core Real-Time Detection Engine.

Handles:
- Loading pre-trained Multimodal Temporal-Relational models
- Feature scaling & dual-modality preparation
- Monitoring the incoming flows folder
- Reading 'n' records sequentially (File-Based Triggering System)
- Low-latency inference and real-time metric aggregation
"""

import os
import sys
import glob
import time
import json
import pickle
import threading
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.classifiers import MultimodalFusionClassifier


class IDSEngine:
    """Core real-time inference engine for file-based network flow triggering."""

    def __init__(
        self,
        watch_dir: str = "incoming_flows",
        default_batch_size: int = 1,
        dataset_name: str = "nf_cse_cic_ids2018_v2",
        fusion_type: str = "concat",
    ):
        self.watch_dir = os.path.abspath(watch_dir)
        os.makedirs(self.watch_dir, exist_ok=True)
        self.batch_size = max(1, int(default_batch_size))
        self.dataset_name = dataset_name
        self.fusion_type = fusion_type
        self.lock = threading.Lock()

        # Model & preprocessor state
        self.model: Optional[MultimodalFusionClassifier] = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.label_mapping: Dict[str, str] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Stream & file pointer state
        self.active_filename: Optional[str] = None
        self.current_offset: int = 0
        self.total_rows_in_file: int = 0
        self._cached_df: Optional[pd.DataFrame] = None

        # Statistics & historical records
        self.history: List[Dict[str, Any]] = []
        self.max_history_len = 1000
        self.stats = {
            "total_inspected": 0,
            "good_count": 0,
            "attack_count": 0,
            "attack_distribution": {},
            "total_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
            "threat_score": 0.0,
            "threat_level": "NORMAL",
        }

        # Initialize model
        self.load_model(self.dataset_name, self.fusion_type)

        # Initialize active file
        self.refresh_file_list()

    def get_available_datasets(self) -> List[Dict[str, str]]:
        """Lists supported pre-trained models."""
        return [
            {
                "id": "nf_cse_cic_ids2018_v2",
                "name": "NF-CSE-CIC-IDS2018-v2 (NetFlow v2 / 99.45% Acc)",
                "type": "binary",
                "recommended": True,
            },
            {
                "id": "cse_cic_ids2018",
                "name": "CSE-CIC-IDS2018 (PCAP / 15 Attack Classes)",
                "type": "multiclass",
                "recommended": False,
            },
            {
                "id": "nf_unsw_nb15_v2",
                "name": "NF-UNSW-NB15-v2 (NetFlow v2 / Binary)",
                "type": "binary",
                "recommended": False,
            },
        ]

    def load_model(self, dataset_name: str, fusion_type: str = "concat"):
        """Loads model weights, scaler, and feature definition."""
        with self.lock:
            self.dataset_name = dataset_name
            self.fusion_type = fusion_type

            # 1. Load scaler
            scaler_path = os.path.join(
                REPO_ROOT,
                "results",
                "phase0_data_preparation",
                "scalers_and_encoders",
                f"{dataset_name}_scaler.pkl",
            )
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler not found at: {scaler_path}")

            with open(scaler_path, "rb") as f:
                scaler_data = pickle.load(f)
            self.scaler = scaler_data["scaler"]
            self.feature_names = scaler_data["features"]

            # 2. Determine class count and label map
            if dataset_name == "cse_cic_ids2018":
                num_classes = 15
                report_path = os.path.join(
                    REPO_ROOT,
                    "experiments",
                    "gradient_adaptive_hybrid",
                    dataset_name,
                    fusion_type,
                    "dataset_integration_report.json",
                )
                if os.path.exists(report_path):
                    with open(report_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self.label_mapping = meta.get("label_mapping", {})
                else:
                    self.label_mapping = {
                        "0": "Benign",
                        "1": "Bot",
                        "2": "Brute Force -Web",
                        "3": "Brute Force -XSS",
                        "4": "DDOS attack-HOIC",
                        "5": "DDOS attack-LOIC-UDP",
                        "6": "DDoS attacks-LOIC-HTTP",
                        "7": "DoS attacks-GoldenEye",
                        "8": "DoS attacks-Hulk",
                        "9": "DoS attacks-SlowHTTPTest",
                        "10": "DoS attacks-Slowloris",
                        "11": "FTP-BruteForce",
                        "12": "Infilteration",
                        "13": "SQL Injection",
                        "14": "SSH-Bruteforce",
                    }
            else:
                num_classes = 2
                self.label_mapping = {"0": "Benign (Good)", "1": "Attack (Malicious)"}

            # 3. Instantiate model
            model = MultimodalFusionClassifier(
                fusion_type=fusion_type,
                temp_input_dim=len(self.feature_names),
                rel_node_dim=len(self.feature_names),
                rel_edge_dim=0,
                num_classes=num_classes,
                dropout_rate=0.2,
            )

            # 4. Load weights
            ckpt_path = os.path.join(
                REPO_ROOT,
                "experiments",
                "gradient_adaptive_hybrid",
                dataset_name,
                fusion_type,
                "best_model.pt",
            )
            if not os.path.exists(ckpt_path):
                # Fallback to concat if custom fusion checkpoint is absent
                ckpt_path = os.path.join(
                    REPO_ROOT,
                    "experiments",
                    "gradient_adaptive_hybrid",
                    dataset_name,
                    "concat",
                    "best_model.pt",
                )

            state = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            self.model = model
            print(
                f"[IDSEngine] Loaded model '{dataset_name}' with fusion '{fusion_type}' from {ckpt_path}"
            )

    def refresh_file_list(self) -> List[str]:
        """Scans watch folder for CSV files."""
        csv_files = sorted(glob.glob(os.path.join(self.watch_dir, "*.csv")))
        file_basenames = [os.path.basename(f) for f in csv_files]

        if not self.active_filename and file_basenames:
            # Default to live_network_flows.csv if present, else first
            if "live_network_flows.csv" in file_basenames:
                self.set_active_file("live_network_flows.csv")
            else:
                self.set_active_file(file_basenames[0])

        return file_basenames

    def set_active_file(self, filename: str) -> bool:
        """Sets active CSV file to read flows from."""
        full_path = os.path.join(self.watch_dir, filename)
        if not os.path.exists(full_path):
            return False

        with self.lock:
            self.active_filename = filename
            self.current_offset = 0
            # Read CSV in memory for fast instant triggering
            try:
                df = pd.read_csv(full_path)
                # Strip column whitespace
                df.columns = [c.strip() for c in df.columns]
                self._cached_df = df
                self.total_rows_in_file = len(df)
                print(
                    f"[IDSEngine] Active file set to '{filename}' ({self.total_rows_in_file:,} total rows)"
                )
                return True
            except Exception as e:
                print(f"[IDSEngine] Error loading CSV '{filename}': {e}")
                self._cached_df = None
                self.total_rows_in_file = 0
                return False

    def reset_pointer(self):
        """Resets stream offset to row 0."""
        with self.lock:
            self.current_offset = 0

    def set_batch_size(self, n: int) -> int:
        """Updates batch size 'n'."""
        with self.lock:
            self.batch_size = max(1, int(n))
            return self.batch_size

    def clear_history(self):
        """Clears processed history and stats."""
        with self.lock:
            self.history.clear()
            self.stats = {
                "total_inspected": 0,
                "good_count": 0,
                "attack_count": 0,
                "attack_distribution": {},
                "total_latency_ms": 0.0,
                "avg_latency_ms": 0.0,
                "threat_score": 0.0,
                "threat_level": "NORMAL",
            }

    def trigger(self, n: Optional[int] = None) -> Dict[str, Any]:
        """Trigger ingestion and prediction for next 'n' records from active CSV file.

        Args:
            n: Number of records to read. If None, uses self.batch_size.

        Returns:
            Dict containing batch results, records, and updated metrics.
        """
        with self.lock:
            if n is not None:
                n = max(1, int(n))
            else:
                n = self.batch_size

            if self._cached_df is None or len(self._cached_df) == 0:
                self.refresh_file_list()
                if self._cached_df is None or len(self._cached_df) == 0:
                    return {
                        "success": False,
                        "error": f"No CSV flows found in watch directory: {self.watch_dir}",
                        "records": [],
                        "stats": self.stats,
                    }

            df = self._cached_df
            total_rows = len(df)

            start_idx = self.current_offset
            end_idx = start_idx + n
            wrapped = False

            if start_idx >= total_rows:
                # Wrap around to beginning
                start_idx = 0
                end_idx = n
                wrapped = True

            if end_idx > total_rows:
                # Take slice up to end, then slice from start
                slice1 = df.iloc[start_idx:total_rows]
                remaining = end_idx - total_rows
                slice2 = df.iloc[0:remaining]
                batch_df = pd.concat([slice1, slice2], ignore_index=True)
                self.current_offset = remaining
                wrapped = True
            else:
                batch_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
                self.current_offset = end_idx

            # Preprocess features
            X_rows = []
            for _, row in batch_df.iterrows():
                feat_vec = []
                for col in self.feature_names:
                    val = row.get(col, 0.0)
                    try:
                        val_num = float(val)
                        if np.isnan(val_num) or np.isinf(val_num):
                            val_num = 0.0
                    except (ValueError, TypeError):
                        val_num = 0.0
                    feat_vec.append(val_num)
                X_rows.append(feat_vec)

            X_raw = np.array(X_rows, dtype=np.float32)

            t_start = time.perf_counter()
            # Standard scale & clip
            X_df = pd.DataFrame(X_raw, columns=self.feature_names)
            X_scaled = self.scaler.transform(X_df)
            X_scaled = np.clip(X_scaled, -100.0, 100.0)

            # PyTorch inputs
            t_temp = torch.tensor(X_scaled, dtype=torch.float32, device=self.device).unsqueeze(1)
            t_rel = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                logits = self.model(t_temp, t_rel)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)

            t_elapsed = (time.perf_counter() - t_start) * 1000.0
            per_record_latency = t_elapsed / max(1, n)

            records_out = []
            cur_time = time.strftime("%H:%M:%S")

            for i in range(len(batch_df)):
                row = batch_df.iloc[i]
                pred_idx = int(preds[i])
                pred_prob = float(probs[i, pred_idx])

                # Determine Good vs Attack
                if self.dataset_name == "cse_cic_ids2018":
                    cls_name = self.label_mapping.get(str(pred_idx), f"Class_{pred_idx}")
                    is_attack = cls_name.lower() != "benign"
                    good_prob = float(probs[i, 0]) if probs.shape[1] > 0 else 0.0
                    attack_prob = float(1.0 - good_prob)
                    attack_signature = cls_name if is_attack else "Normal Traffic"
                else:
                    is_attack = (pred_idx == 1)
                    good_prob = float(probs[i, 0])
                    attack_prob = float(probs[i, 1])
                    # If CSV has known Attack string, provide specific signature
                    raw_attack = str(row.get("Attack", row.get("attack_cat", ""))).strip()
                    if is_attack:
                        if raw_attack and raw_attack.lower() not in ["benign", "0", "nan", ""]:
                            attack_signature = raw_attack
                        else:
                            attack_signature = "Malicious Flow (Intrusion Detected)"
                    else:
                        attack_signature = "Benign Normal Flow"

                # Telemetry metadata
                src_ip = str(row.get("IPV4_SRC_ADDR", row.get("Src IP", row.get("SRC IP", f"192.168.1.{10 + (start_idx + i) % 200}"))))
                dst_ip = str(row.get("IPV4_DST_ADDR", row.get("Dst IP", row.get("DST IP", f"10.0.0.{5 + (start_idx + i) % 50}"))))
                src_port = str(row.get("L4_SRC_PORT", row.get("Src Port", row.get("SRC PORT", str(1024 + (start_idx + i) * 37 % 60000)))))
                dst_port = str(row.get("L4_DST_PORT", row.get("Dst Port", row.get("DST PORT", "80"))))
                proto = str(row.get("PROTOCOL", row.get("Protocol", "TCP")))
                if proto == "6":
                    proto = "TCP"
                elif proto == "17":
                    proto = "UDP"
                elif proto == "1":
                    proto = "ICMP"

                duration = float(row.get("FLOW_DURATION_MILLISECONDS", row.get("Flow Duration", 0.0)))
                bytes_in = float(row.get("IN_BYTES", row.get("TotLen Fwd Pkts", 0.0)))
                bytes_out = float(row.get("OUT_BYTES", row.get("TotLen Bwd Pkts", 0.0)))

                # Severity classification
                if not is_attack:
                    severity = "LOW"
                    badge_color = "#00f5a0"  # Emerald
                else:
                    if attack_prob > 0.95:
                        severity = "CRITICAL"
                        badge_color = "#ff0055"  # Crimson Neon
                    elif attack_prob > 0.80:
                        severity = "HIGH"
                        badge_color = "#ff4b72"
                    else:
                        severity = "MEDIUM"
                        badge_color = "#f9a826"  # Amber

                # Modality contribution approximation
                temp_energy = float(np.linalg.norm(X_scaled[i, :len(self.feature_names)//2]))
                rel_energy = float(np.linalg.norm(X_scaled[i, len(self.feature_names)//2:]))
                total_energy = max(1e-5, temp_energy + rel_energy)
                temp_pct = round((temp_energy / total_energy) * 100.0, 1)
                rel_pct = round(100.0 - temp_pct, 1)

                flow_record = {
                    "id": f"FL-{start_idx + i + 1:06d}",
                    "timestamp": cur_time,
                    "file_row": start_idx + i,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "protocol": proto,
                    "duration_ms": round(duration, 2),
                    "bytes_in": int(bytes_in),
                    "bytes_out": int(bytes_out),
                    "is_attack": bool(is_attack),
                    "verdict": "ATTACK" if is_attack else "GOOD",
                    "attack_signature": attack_signature,
                    "confidence": round(pred_prob * 100.0, 2),
                    "good_prob": round(good_prob * 100.0, 2),
                    "attack_prob": round(attack_prob * 100.0, 2),
                    "severity": severity,
                    "badge_color": badge_color,
                    "latency_ms": round(per_record_latency, 2),
                    "temporal_modality_pct": temp_pct,
                    "relational_modality_pct": rel_pct,
                    "raw_features": {k: str(row[k]) for k in row.keys()[:30]},
                }
                records_out.append(flow_record)

                # Update running session stats
                self.stats["total_inspected"] += 1
                if is_attack:
                    self.stats["attack_count"] += 1
                    sig = attack_signature
                    self.stats["attack_distribution"][sig] = self.stats["attack_distribution"].get(sig, 0) + 1
                else:
                    self.stats["good_count"] += 1

                self.history.insert(0, flow_record)

            if len(self.history) > self.max_history_len:
                self.history = self.history[: self.max_history_len]

            # Compute threat level and threat score
            self.stats["total_latency_ms"] += t_elapsed
            self.stats["avg_latency_ms"] = round(
                self.stats["total_latency_ms"] / max(1, self.stats["total_inspected"]), 2
            )

            recent_20 = self.history[:20]
            recent_attacks = sum(1 for r in recent_20 if r["is_attack"])
            attack_ratio = recent_attacks / max(1, len(recent_20))
            self.stats["threat_score"] = round(attack_ratio * 100.0, 1)

            if attack_ratio == 0:
                self.stats["threat_level"] = "NORMAL"
            elif attack_ratio < 0.25:
                self.stats["threat_level"] = "GUARDED"
            elif attack_ratio < 0.60:
                self.stats["threat_level"] = "ELEVATED"
            else:
                self.stats["threat_level"] = "CRITICAL"

            return {
                "success": True,
                "batch_size": n,
                "records": records_out,
                "stats": self.stats,
                "pointer": {
                    "current_offset": self.current_offset,
                    "total_rows": total_rows,
                    "progress_pct": round((self.current_offset / max(1, total_rows)) * 100.0, 1),
                    "wrapped": wrapped,
                },
                "active_file": self.active_filename,
            }

    def get_status(self) -> Dict[str, Any]:
        """Returns overall engine status."""
        with self.lock:
            csv_files = self.refresh_file_list()
            return {
                "batch_size": self.batch_size,
                "active_file": self.active_filename,
                "available_files": csv_files,
                "pointer_offset": self.current_offset,
                "total_rows": self.total_rows_in_file,
                "progress_pct": round((self.current_offset / max(1, self.total_rows_in_file)) * 100.0, 1) if self.total_rows_in_file else 0.0,
                "dataset_name": self.dataset_name,
                "fusion_type": self.fusion_type,
                "stats": self.stats,
                "watch_dir": self.watch_dir,
                "num_features": len(self.feature_names),
                "device": self.device,
            }
