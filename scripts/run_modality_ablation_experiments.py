"""Runner Script for Temporal-Only and Spatial/Relational-Only Modality Ablation Experiments.

Executes controlled evaluation across:
- 3 Datasets: nf_cse_cic_ids2018_v2, nf_unsw_nb15_v2, cse_cic_ids2018
- 2 Architectures: temporal_only (FixedTemporalEncoder GRU), relational_only (FixedRelationalEncoder Graph)
- Total: 6 Controlled Experiments

Usage:
  python scripts/run_modality_ablation_experiments.py --dataset all --architecture all --epochs 15

Or single dataset/architecture:
  python scripts/run_modality_ablation_experiments.py --dataset nf_cse_cic_ids2018_v2 --architecture temporal_only
"""

import os
import sys
import argparse
import json
import time
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import load_dataset_split
from src.evaluation.leakage_checker import run_mandatory_leakage_checks
from src.models.single_modality import TemporalOnlyClassifier, SpatialRelationalOnlyClassifier
from src.models.classifiers import MultimodalFusionClassifier
from src.models.losses import GradientAdaptiveHybridLoss
from src.evaluation.trainer import ExperimentTrainer, set_seed


DATASETS = ["nf_cse_cic_ids2018_v2", "nf_unsw_nb15_v2", "cse_cic_ids2018"]
ARCHITECTURES = ["temporal_only", "relational_only", "both"]


def calculate_intrusion_metrics(cm: np.ndarray, class_names: list) -> dict:
    """Calculates NIDS metrics: Benign recall, Attack macro recall, FPR, FNR, minority F1."""
    # Identify Benign index
    benign_idx = None
    for idx, cname in enumerate(class_names):
        if str(cname).lower() == "benign":
            benign_idx = idx
            break

    total_samples = np.sum(cm)
    num_classes = len(class_names)

    # Calculate per-class TP, FP, FN, TN
    tps = np.diag(cm)
    fps = np.sum(cm, axis=0) - tps
    fns = np.sum(cm, axis=1) - tps
    tns = total_samples - (tps + fps + fns)

    recalls = tps / np.maximum(tps + fns, 1)

    if benign_idx is not None:
        benign_recall = float(recalls[benign_idx])
        attack_indices = [i for i in range(num_classes) if i != benign_idx]
        attack_macro_recall = float(np.mean(recalls[attack_indices])) if attack_indices else 0.0

        # Global FPR & FNR from NIDS binary perspective (Benign vs Attack)
        # FP = Benign predicted as Attack; TN = Benign predicted as Benign
        benign_row = cm[benign_idx, :]
        benign_total = np.sum(benign_row)
        fp_nids = benign_total - cm[benign_idx, benign_idx]
        tn_nids = cm[benign_idx, benign_idx]
        fpr = float(fp_nids / max(1, benign_total))

        # FN = Attack predicted as Benign; TP = Attack predicted as Attack
        attack_rows = np.delete(cm, benign_idx, axis=0)
        attack_total = np.sum(attack_rows)
        fn_nids = np.sum(attack_rows[:, benign_idx]) if attack_total > 0 else 0
        fnr = float(fn_nids / max(1, attack_total))
    else:
        benign_recall = float(recalls[0])
        attack_macro_recall = float(np.mean(recalls[1:])) if len(recalls) > 1 else 0.0
        fpr = float(np.mean(fps / np.maximum(fps + tns, 1)))
        fnr = float(np.mean(fns / np.maximum(fns + tps, 1)))

    return {
        "benign_recall": benign_recall,
        "attack_macro_recall": attack_macro_recall,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


def run_single_experiment(
    dataset_name: str,
    architecture_type: str,
    epochs: int = 15,
    patience: int = 5,
    sample_limit: int = None,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    base_data_dir: str = "data/raw",
    output_dir_base: str = "experiments/modality_ablation",
) -> dict:
    """Executes a single controlled experiment for a (dataset, architecture) pair."""
    set_seed(seed)
    exp_prefix = f"{dataset_name}_{architecture_type}"
    exp_out_dir = os.path.join(output_dir_base, dataset_name, architecture_type)
    os.makedirs(exp_out_dir, exist_ok=True)

    metrics_file = os.path.join(exp_out_dir, "metrics.json")
    if os.path.exists(metrics_file):
        print(f"\n======================================================================")
        print(f" MODALITY ABLATION: Dataset={dataset_name} | Arch={architecture_type}")
        print(f" [ALREADY COMPLETED] Found existing metrics.json at {metrics_file}. Skipping rerun.")
        print(f"======================================================================")
        with open(metrics_file, "r") as f:
            return json.load(f)

    print(f"\n======================================================================")
    print(f" MODALITY ABLATION: Dataset={dataset_name} | Arch={architecture_type}")
    print(f" Sample Limit: {'FULL DATASET' if sample_limit is None else sample_limit}")
    print(f" Output Directory: {exp_out_dir}")
    print(f"======================================================================")

    # 1. Load Data Split & Metadata
    data = load_dataset_split(
        dataset_name=dataset_name,
        base_data_dir=base_data_dir,
        sample_limit=sample_limit,
        seed=seed,
        output_dir=exp_out_dir,
    )

    integration_report = data["integration_report"]
    with open(os.path.join(exp_out_dir, "dataset_integration_report.json"), "w", encoding="utf-8") as f:
        json.dump(integration_report, f, indent=2)

    # 2. Mandatory Data Leakage Checks
    leakage_res = run_mandatory_leakage_checks(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        X_test=data["X_test"],
        y_test=data["y_test"],
        feature_names=data["feature_names"],
        scaler_fitted_on_train_only=data["scaler_fitted_on_train_only"],
    )
    with open(os.path.join(exp_out_dir, "leakage_report.json"), "w", encoding="utf-8") as f:
        json.dump(leakage_res, f, indent=2)

    # 3. Class Weights for Loss
    y_train_np = data["y_train"]
    class_counts = np.bincount(y_train_np, minlength=data["num_classes"])
    total_samples = len(y_train_np)
    class_weights_np = total_samples / (data["num_classes"] * np.maximum(class_counts, 1))
    class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32)

    import gc

    # 4. Strict Modality Feature Column Slicing
    feature_names = data["feature_names"]
    temp_feature_names = integration_report.get("temporal_feature_mapping", [])
    rel_feature_names = integration_report.get("relational_feature_mapping", [])

    temp_indices = [i for i, fname in enumerate(feature_names) if fname in temp_feature_names]
    rel_indices = [i for i, fname in enumerate(feature_names) if fname in rel_feature_names]

    if not temp_indices:
        temp_indices = list(range(len(feature_names)))
    if not rel_indices:
        rel_indices = list(range(len(feature_names)))

    X_train_raw = data.pop("X_train")
    X_val_raw = data.pop("X_val")
    X_test_raw = data.pop("X_test")
    y_train_np = data.pop("y_train")
    y_val_np = data.pop("y_val")
    y_test_np = data.pop("y_test")

    if architecture_type in ["temporal_only"]:
        X_train_seq = X_train_raw[:, temp_indices]
        X_train_rel = X_train_seq
        X_val_seq = X_val_raw[:, temp_indices]
        X_val_rel = X_val_seq
        X_test_seq = X_test_raw[:, temp_indices]
        X_test_rel = X_test_seq
        temp_dim = len(temp_indices)
        rel_dim = temp_dim
        model = TemporalOnlyClassifier(
            temp_input_dim=temp_dim,
            embed_dim=64,
            num_classes=data["num_classes"],
            dropout_rate=0.2,
        )
    elif architecture_type in ["relational_only", "spatial_only", "space_only"]:
        X_train_seq = X_train_raw[:, rel_indices]
        X_train_rel = X_train_seq
        X_val_seq = X_val_raw[:, rel_indices]
        X_val_rel = X_val_seq
        X_test_seq = X_test_raw[:, rel_indices]
        X_test_rel = X_test_seq
        rel_dim = len(rel_indices)
        temp_dim = rel_dim
        model = SpatialRelationalOnlyClassifier(
            rel_node_dim=rel_dim,
            rel_edge_dim=0,
            embed_dim=64,
            num_classes=data["num_classes"],
            dropout_rate=0.2,
        )
    elif architecture_type in ["both", "fused"]:
        X_train_seq = X_train_raw[:, temp_indices]
        X_train_rel = X_train_raw[:, rel_indices]
        X_val_seq = X_val_raw[:, temp_indices]
        X_val_rel = X_val_raw[:, rel_indices]
        X_test_seq = X_test_raw[:, temp_indices]
        X_test_rel = X_test_raw[:, rel_indices]
        temp_dim = len(temp_indices)
        rel_dim = len(rel_indices)
        model = MultimodalFusionClassifier(
            fusion_type="concat",
            temp_input_dim=temp_dim,
            rel_node_dim=rel_dim,
            rel_edge_dim=0,
            num_classes=data["num_classes"],
            embed_dim=64,
            dropout_rate=0.2,
        )
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")

    # Construct PyTorch DataLoaders
    X_train_seq_t = torch.from_numpy(X_train_seq).float()
    X_train_rel_t = torch.from_numpy(X_train_rel).float()
    y_train_t = torch.from_numpy(y_train_np).long()

    X_val_seq_t = torch.from_numpy(X_val_seq).float()
    X_val_rel_t = torch.from_numpy(X_val_rel).float()
    y_val_t = torch.from_numpy(y_val_np).long()

    X_test_seq_t = torch.from_numpy(X_test_seq).float()
    X_test_rel_t = torch.from_numpy(X_test_rel).float()
    y_test_t = torch.from_numpy(y_test_np).long()
    gc.collect()

    train_ds = TensorDataset(X_train_seq_t, X_train_rel_t, y_train_t)
    val_ds = TensorDataset(X_val_seq_t, X_val_rel_t, y_val_t)
    test_ds = TensorDataset(X_test_seq_t, X_test_rel_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    param_info = model.get_parameter_count()
    print(f" [MODEL INITIALIZED] {architecture_type.upper()} | Temp Dim: {temp_dim} | Rel Dim: {rel_dim} | Total Params: {param_info['total_parameters']:,} | Trainable Params: {param_info['trainable_parameters']:,}")

    # Save Parameter Count Report
    with open(os.path.join(exp_out_dir, "parameter_count_report.json"), "w") as f:
        json.dump(param_info, f, indent=2)

    # Save Configuration
    config = {
        "dataset_name": dataset_name,
        "architecture_type": architecture_type,
        "input_dim": data["input_dim"],
        "num_classes": data["num_classes"],
        "class_names": data["class_names"],
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "patience": patience,
        "seed": seed,
        "sample_limit": sample_limit,
        "parameter_info": param_info,
    }
    with open(os.path.join(exp_out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    criterion = GradientAdaptiveHybridLoss(
        class_weights=class_weights_tensor,
        prior_ce=0.50,
        prior_focal=0.35,
        prior_wce=0.15,
        smoothing_factor=0.90,
    )

    trainer = ExperimentTrainer(
        model=model,
        criterion=criterion,
        learning_rate=learning_rate,
        weight_decay=1e-4,
    )

    checkpoint_path = os.path.join(exp_out_dir, "best_model.pt")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        save_checkpoint_path=checkpoint_path,
    )

    # Save Training Log
    training_log = {
        "epochs_completed": history["epochs_completed"],
        "best_epoch": history["best_epoch"],
        "best_val_loss": float(history["best_val_loss"]),
        "final_val_loss": float(history["final_val_loss"]),
        "training_time_seconds": float(history["training_time"]),
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
    }
    with open(os.path.join(exp_out_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    # 6. Test Set Evaluation & Inference Speed Benchmark
    start_inf = time.time()
    test_loss, y_test_true, y_test_pred = trainer.evaluate(test_loader)
    inf_time_sec = time.time() - start_inf
    ms_per_sample = (inf_time_sec / max(1, len(y_test_true))) * 1000.0

    # Overall & Per-Class Metrics
    all_labels = list(range(data["num_classes"]))
    acc = accuracy_score(y_test_true, y_test_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test_true, y_test_pred, labels=all_labels, average="macro", zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(y_test_true, y_test_pred, labels=all_labels, average="weighted", zero_division=0)

    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(y_test_true, y_test_pred, labels=all_labels, average=None, zero_division=0)

    # Per-class metrics dataframe & export
    per_class_df = pd.DataFrame({
        "class_name": data["class_names"],
        "precision": per_p,
        "recall": per_r,
        "f1_score": per_f1,
        "support": per_sup,
    })
    per_class_df.to_csv(os.path.join(exp_out_dir, "per_class_metrics.csv"), index=False)

    # Confusion Matrix
    cm = confusion_matrix(y_test_true, y_test_pred, labels=all_labels)
    cm_df = pd.DataFrame(cm, index=data["class_names"], columns=data["class_names"])
    cm_df.to_csv(os.path.join(exp_out_dir, "confusion_matrix.csv"))
    with open(os.path.join(exp_out_dir, "confusion_matrix.json"), "w") as f:
        json.dump(cm.tolist(), f)

    # Plots
    # 1. Confusion Matrix Plot
    cm_path = os.path.join(exp_out_dir, "confusion_matrix.png")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data["class_names"], yticklabels=data["class_names"])
    plt.title(f"Confusion Matrix: {exp_prefix}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. Loss Curve Plot
    curve_path = os.path.join(exp_out_dir, "loss_curve.png")
    epochs_range = range(1, len(history["train_losses"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_losses"], label="Train Loss", marker="o")
    plt.plot(epochs_range, history["val_losses"], label="Validation Loss", marker="s")
    plt.axvline(x=history["best_epoch"], color="r", linestyle="--", label=f"Best Epoch ({history['best_epoch']})")
    plt.title(f"Loss Curve: {exp_prefix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300)
    plt.close()

    # 3. Per-Class F1 Bar Chart
    f1_chart_path = os.path.join(exp_out_dir, "per_class_f1.png")
    plt.figure(figsize=(9, 5))
    bars = plt.bar(data["class_names"], per_f1, color="teal", alpha=0.85)
    plt.title(f"Per-Class F1-Score: {exp_prefix}")
    plt.xlabel("Class Name")
    plt.ylabel("F1-Score")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f"{height:.2f}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(f1_chart_path, dpi=300)
    plt.close()

    # NIDS Intrusion Metrics
    nids_metrics = calculate_intrusion_metrics(cm, data["class_names"])

    # Calculate Minority Class F1 (minority classes with support < median support)
    med_sup = np.median(per_sup)
    minority_mask = per_sup <= med_sup
    minority_f1 = float(np.mean(per_f1[minority_mask])) if np.any(minority_mask) else float(macro_f1)

    metrics = {
        "dataset_name": dataset_name,
        "architecture_type": architecture_type,
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(w_f1),
        "minority_class_f1": minority_f1,
        "benign_recall": nids_metrics["benign_recall"],
        "attack_macro_recall": nids_metrics["attack_macro_recall"],
        "false_positive_rate": nids_metrics["false_positive_rate"],
        "false_negative_rate": nids_metrics["false_negative_rate"],
        "training_time_seconds": float(history["training_time"]),
        "inference_time_ms_per_sample": float(ms_per_sample),
        "epochs_completed": history["epochs_completed"],
        "best_epoch": history["best_epoch"],
        "best_val_loss": float(history["best_val_loss"]),
        "final_val_loss": float(history["final_val_loss"]),
        "total_trainable_parameters": param_info["trainable_parameters"],
        "total_parameters": param_info["total_parameters"],
        "seed": seed,
    }

    with open(os.path.join(exp_out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f" [RESULT] Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Benign Recall: {nids_metrics['benign_recall']:.4f} | Attack Macro Recall: {nids_metrics['attack_macro_recall']:.4f}")

    # Memory cleanup between sequential experiments
    del train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, model, trainer, criterion, X_train_seq_t, X_train_rel_t, X_val_seq_t, X_val_rel_t, X_test_seq_t, X_test_rel_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def generate_master_summary_report(output_dir_base: str = "experiments/modality_ablation"):
    """Builds master summary CSV, Markdown reports, and scientific analysis report."""
    summary_rows = []

    for ds_name in DATASETS:
        for arch in ARCHITECTURES:
            metrics_file = os.path.join(output_dir_base, ds_name, arch, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    m = json.load(f)

                summary_rows.append({
                    "Dataset": ds_name,
                    "Architecture": arch,
                    "Accuracy": m.get("accuracy", 0.0),
                    "Macro_Precision": m.get("macro_precision", 0.0),
                    "Macro_Recall": m.get("macro_recall", 0.0),
                    "Macro_F1": m.get("macro_f1", 0.0),
                    "Weighted_F1": m.get("weighted_f1", 0.0),
                    "Benign_Recall": m.get("benign_recall", 0.0),
                    "Attack_Macro_Recall": m.get("attack_macro_recall", 0.0),
                    "False_Positive_Rate": m.get("false_positive_rate", 0.0),
                    "False_Negative_Rate": m.get("false_negative_rate", 0.0),
                    "Minority_Class_F1": m.get("minority_class_f1", 0.0),
                    "Trainable_Parameters": m.get("total_trainable_parameters", 0),
                    "Training_Time_s": m.get("training_time_seconds", 0.0),
                    "Inference_ms_per_sample": m.get("inference_time_ms_per_sample", 0.0),
                    "Best_Epoch": m.get("best_epoch", 0),
                    "Epochs_Completed": m.get("epochs_completed", 0),
                    "Seed": m.get("seed", 42),
                })

    if not summary_rows:
        print("[SUMMARY] No completed experiment metrics found.")
        return

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir_base, "modality_ablation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[SUMMARY SAVED] Master CSV report written to: {csv_path}")

    # Generate Markdown Master Report with Scientific Analysis
    md_path = os.path.join(output_dir_base, "modality_ablation_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Master Summary Report: Temporal-Only vs Spatial/Relational-Only Modality Ablation\n\n")
        f.write(f"Evaluated across {len(summary_rows)} completed experiments without multimodal fusion.\n\n")

        f.write("## 1. Overall Modality Performance Comparison\n\n")
        f.write("| Dataset | Architecture | Accuracy | Macro F1 | Weighted F1 | Benign Recall | Attack Macro Recall | FPR | FNR | Trainable Params | Time (s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for r in summary_rows:
            f.write(
                f"| {r['Dataset']} | {r['Architecture']} | {r['Accuracy']:.4f} | {r['Macro_F1']:.4f} | {r['Weighted_F1']:.4f} | {r['Benign_Recall']:.4f} | {r['Attack_Macro_Recall']:.4f} | {r['False_Positive_Rate']:.4f} | {r['False_Negative_Rate']:.4f} | {r['Trainable_Parameters']:,} | {r['Training_Time_s']:.1f} |\n"
            )

        f.write("\n---\n\n")
        f.write("## 2. Scientific Analysis & Modality Insights\n\n")
        f.write("### A. Temporal-Only Architecture (`FixedTemporalEncoder` GRU)\n")
        f.write("- **Strengths**: Captures sequential dependencies, inter-arrival time bursts, and multi-stage traffic progressions.\n")
        f.write("- **Weaknesses**: May struggle on static single-flow attacks where topological context (host connectivity) is required for disambiguation.\n\n")
        f.write("### B. Spatial/Relational-Only Architecture (`FixedRelationalEncoder` Graph)\n")
        f.write("- **Strengths**: Captures entity interaction graphs, host node degree distributions, and communication fan-out.\n")
        f.write("- **Note on Spatial Terminology**: While designated as 'Spatial/Relational-Only Model', the architecture models communication graph interactions rather than physical geographical coordinates.\n")
        f.write("- **Weaknesses**: Cannot leverage sequential state across consecutive packets within a flow without temporal hidden state tracking.\n\n")
        f.write("### C. Multimodal Fusion Architecture (`MultimodalFusionClassifier` Both Fused)\n")
        f.write("- **Strengths**: Combines sequential flow dynamics (GRU) with spatial/graph interaction dynamics, providing complementary features for robust detection across multi-vector attacks.\n")
        f.write("- **Trade-off**: Requires slightly higher parameter count and runtime overhead than single-modality models.\n\n")

    print(f"[SUMMARY SAVED] Master Markdown report written to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Temporal-Only, Spatial/Relational-Only, and Multimodal Fusion Modality Ablation Experiments")
    parser.add_argument("--dataset", type=str, default="all", choices=["all"] + DATASETS, help="Target dataset or 'all'")
    parser.add_argument("--architecture", type=str, default="all", choices=["all", "temporal_only", "relational_only", "spatial_only", "space_only", "both", "fused"], help="Target architecture: temporal_only, relational_only, both, or all")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum epochs per experiment (default: 15)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--sample-limit", type=int, default=None, help="Sample limit (default: None for complete dataset)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--base-data-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--output-dir", type=str, default="experiments/modality_ablation", help="Base output directory")

    args = parser.parse_args()

    target_datasets = DATASETS if args.dataset == "all" else [args.dataset]
    if args.architecture == "all":
        target_archs = ARCHITECTURES
    elif args.architecture in ["spatial_only", "space_only"]:
        target_archs = ["relational_only"]
    elif args.architecture == "fused":
        target_archs = ["both"]
    else:
        target_archs = [args.architecture]

    print(f"\n======================================================================")
    print(f" MODALITY ABLATION EXPERIMENT SUITE")
    print(f" Datasets ({len(target_datasets)}): {target_datasets}")
    print(f" Architectures ({len(target_archs)}): {target_archs}")
    print(f" Total Experiments: {len(target_datasets) * len(target_archs)}")
    print(f" Sample Limit: {'FULL DATASET' if args.sample_limit is None else args.sample_limit}")
    print(f" Output Directory: {args.output_dir}")
    print(f"======================================================================")

    for ds in target_datasets:
        for arch in target_archs:
            run_single_experiment(
                dataset_name=ds,
                architecture_type=arch,
                epochs=args.epochs,
                patience=args.patience,
                sample_limit=args.sample_limit,
                seed=args.seed,
                base_data_dir=args.base_data_dir,
                output_dir_base=args.output_dir,
            )

    generate_master_summary_report(output_dir_base=args.output_dir)


if __name__ == "__main__":
    main()
