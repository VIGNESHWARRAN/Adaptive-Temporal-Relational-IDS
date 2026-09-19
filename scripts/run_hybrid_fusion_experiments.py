"""Runner Script for Gradient-Based Adaptive Hybrid Loss + 18 Fusion Experiments.

Executes controlled evaluation across:
- 3 Datasets: nf_cse_cic_ids2018_v2, nf_unsw_nb15_v2, cse_cic_ids2018
- 6 Multimodal Fusion Methods: concat, gated, bilinear, cross_attention, decision_ensemble, quantum_pqc
- Total: 18 Experiments

Usage:
  python scripts/run_hybrid_fusion_experiments.py --dataset all --fusion all --epochs 15 --sample-limit 100000

Or single dataset/fusion:
  python scripts/run_hybrid_fusion_experiments.py --dataset nf_cse_cic_ids2018_v2 --fusion concat
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import load_dataset_split
from src.evaluation.leakage_checker import run_mandatory_leakage_checks
from src.models.classifiers import MultimodalFusionClassifier
from src.models.losses import GradientAdaptiveHybridLoss
from src.evaluation.trainer import ExperimentTrainer, set_seed, compute_metrics_and_plots


DATASETS = ["nf_cse_cic_ids2018_v2", "nf_unsw_nb15_v2", "cse_cic_ids2018"]
FUSION_METHODS = [
    "concat",
    "gated",
    "bilinear",
    "cross_attention",
    "decision_ensemble",
    "quantum_pqc",
]


def run_single_experiment(
    dataset_name: str,
    fusion_type: str,
    epochs: int = 15,
    patience: int = 5,
    sample_limit: int = 100000,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    base_data_dir: str = "data/raw",
    output_dir_base: str = "experiments/gradient_adaptive_hybrid",
) -> dict:
    """Executes a single controlled experiment for a (dataset, fusion) pair."""
    set_seed(seed)
    exp_prefix = f"{dataset_name}_{fusion_type}"
    exp_out_dir = os.path.join(output_dir_base, dataset_name, fusion_type)
    os.makedirs(exp_out_dir, exist_ok=True)

    print(f"\n======================================================================")
    print(f" EXPERIMENT: Dataset={dataset_name} | Fusion={fusion_type}")
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

    if not leakage_res["all_passed"]:
        print(f"[WARN] Leakage check warnings detected for {exp_prefix}. Check leakage_report.json.")

    # 3. Compute Inverse Class Frequency Weights for Loss
    y_train_np = data["y_train"]
    class_counts = np.bincount(y_train_np, minlength=data["num_classes"])
    total_samples = len(y_train_np)
    class_weights_np = total_samples / (data["num_classes"] * np.maximum(class_counts, 1))
    class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32)

    # 4. Construct DataLoaders
    # Note: Dual API passed: sequence tensor + node feature tensor
    X_train_t = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train_t = torch.tensor(data["y_train"], dtype=torch.long)
    X_val_t = torch.tensor(data["X_val"], dtype=torch.float32)
    y_val_t = torch.tensor(data["y_val"], dtype=torch.long)
    X_test_t = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test_t = torch.tensor(data["y_test"], dtype=torch.long)

    train_ds = TensorDataset(X_train_t, X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 5. Instantiate Multimodal Model & Gradient Adaptive Loss
    model = MultimodalFusionClassifier(
        fusion_type=fusion_type,
        temp_input_dim=data["input_dim"],
        rel_node_dim=data["input_dim"],
        rel_edge_dim=0,
        num_classes=data["num_classes"],
        dropout_rate=0.2,
    )

    criterion = GradientAdaptiveHybridLoss(
        class_weights=class_weights_tensor,
        prior_ce=0.50,
        prior_focal=0.35,
        prior_wce=0.15,
        smoothing_factor=0.90,
    )

    # 6. Train Model
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

    # 7. Final Test Set Evaluation
    test_loss, y_test_true, y_test_pred = trainer.evaluate(test_loader)

    metrics = compute_metrics_and_plots(
        y_true=y_test_true,
        y_pred=y_test_pred,
        class_names=data["class_names"],
        output_dir=exp_out_dir,
        exp_prefix=exp_prefix,
        train_history=history,
    )

    print(f" [RESULT] Test Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f}")
    if "adaptive_loss_summary" in metrics:
        diag_sum = metrics["adaptive_loss_summary"]
        print(f" [ADAPTIVE LOSS] Final Weights -> alpha (CE): {diag_sum['final_alpha']:.4f}, beta (Focal): {diag_sum['final_beta']:.4f}, gamma (WCE): {diag_sum['final_gamma']:.4f}")
        print(f" [ADAPTIVE LOSS] Events -> Clipping: {diag_sum['clipping_events']}, Fallback: {diag_sum['fallback_events']}")

    return metrics


def generate_master_summary_report(output_dir_base: str = "experiments/gradient_adaptive_hybrid"):
    """Scans all completed experiment directories and builds master summary CSV & Markdown reports."""
    summary_rows = []

    for ds_name in DATASETS:
        for fusion in FUSION_METHODS:
            metrics_file = os.path.join(output_dir_base, ds_name, fusion, "metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    m = json.load(f)

                diag_sum = m.get("adaptive_loss_summary", {})
                summary_rows.append({
                    "Dataset": ds_name,
                    "Fusion_Method": fusion,
                    "Accuracy": m.get("accuracy", 0.0),
                    "Macro_Precision": m.get("macro_precision", 0.0),
                    "Macro_Recall": m.get("macro_recall", 0.0),
                    "Macro_F1": m.get("macro_f1", 0.0),
                    "Weighted_F1": m.get("weighted_f1", 0.0),
                    "Best_Epoch": m.get("best_epoch", 0),
                    "Epochs_Completed": m.get("epochs_completed", 0),
                    "Training_Time_s": m.get("training_time_seconds", 0.0),
                    "Final_Alpha_CE": diag_sum.get("final_alpha", 0.50),
                    "Final_Beta_Focal": diag_sum.get("final_beta", 0.35),
                    "Final_Gamma_WCE": diag_sum.get("final_gamma", 0.15),
                    "Clipping_Events": diag_sum.get("clipping_events", 0),
                    "Fallback_Events": diag_sum.get("fallback_events", 0),
                })

    if not summary_rows:
        print("[SUMMARY] No completed experiment metrics found.")
        return

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir_base, "master_summary_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[SUMMARY SAVED] Master CSV report written to: {csv_path}")

    # Generate Markdown Table
    md_path = os.path.join(output_dir_base, "master_summary_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Master Summary Report: Gradient-Based Adaptive Hybrid Loss + 18 Fusion Experiments\n\n")
        f.write(f"Generated across {len(summary_rows)} completed experiments.\n\n")
        f.write("| Dataset | Fusion Method | Accuracy | Macro F1 | Weighted F1 | Best Epoch | Time (s) | Final alpha (CE) | Final beta (Focal) | Final gamma (WCE) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

        for r in summary_rows:
            f.write(
                f"| {r['Dataset']} | {r['Fusion_Method']} | {r['Accuracy']:.4f} | {r['Macro_F1']:.4f} | {r['Weighted_F1']:.4f} | {r['Best_Epoch']} | {r['Training_Time_s']:.1f} | {r['Final_Alpha_CE']:.4f} | {r['Final_Beta_Focal']:.4f} | {r['Final_Gamma_WCE']:.4f} |\n"
            )

    print(f"[SUMMARY SAVED] Master Markdown report written to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Gradient-Based Adaptive Hybrid Loss + 18 Fusion Experiments")
    parser.add_argument("--dataset", type=str, default="all", choices=["all"] + DATASETS, help="Target dataset or 'all'")
    parser.add_argument("--fusion", type=str, default="all", choices=["all"] + FUSION_METHODS, help="Target fusion method or 'all'")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum epochs per experiment (default: 15)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--sample-limit", type=int, default=100000, help="Sampling quota limit (default: 100000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--base-data-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--output-dir", type=str, default="experiments/gradient_adaptive_hybrid", help="Base output directory")

    args = parser.parse_args()

    target_datasets = DATASETS if args.dataset == "all" else [args.dataset]
    target_fusions = FUSION_METHODS if args.fusion == "all" else [args.fusion]

    print(f"\n======================================================================")
    print(f" GRADIENT-BASED ADAPTIVE HYBRID LOSS EXPERIMENT SUITE")
    print(f" Datasets ({len(target_datasets)}): {target_datasets}")
    print(f" Fusion Methods ({len(target_fusions)}): {target_fusions}")
    print(f" Total Experiments: {len(target_datasets) * len(target_fusions)}")
    print(f" Output Directory: {args.output_dir}")
    print(f"======================================================================")

    for ds in target_datasets:
        for fusion in target_fusions:
            run_single_experiment(
                dataset_name=ds,
                fusion_type=fusion,
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
