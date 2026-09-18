"""Runner Script for Phase 1 Controlled Loss-Function Experiments.

Usage:
  python scripts/run_phase1_experiments.py --dataset all --loss all --epochs 15 --sample-limit 100000

Or run single experiment:
  python scripts/run_phase1_experiments.py --dataset nf_cse_cic_ids2018_v2 --loss standard_ce
"""

import os
import sys
import argparse
import json
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.loader import load_dataset_split
from src.evaluation.leakage_checker import run_mandatory_leakage_checks
from src.models.classifiers import MultimodalFusionClassifier
from src.models.losses import StandardCrossEntropyLoss, ClassWeightedCrossEntropyLoss, FocalLoss
from src.evaluation.trainer import ExperimentTrainer, set_seed, compute_metrics_and_plots


DATASETS = ["nf_cse_cic_ids2018_v2", "nf_unsw_nb15_v2", "cse_cic_ids2018"]
LOSS_TYPES = ["standard_ce", "weighted_ce", "focal_loss"]


def run_single_experiment(
    dataset_name: str,
    loss_type: str,
    epochs: int = 15,
    sample_limit: int = 100000,
    seed: int = 42,
    base_data_dir: str = "data/raw",
    output_dir_base: str = "results/phase1_loss_experiments",
):
    """Executes a single controlled experiment."""
    set_seed(seed)
    exp_prefix = f"{dataset_name}_{loss_type}"
    exp_out_dir = os.path.join(output_dir_base, dataset_name, loss_type)
    os.makedirs(exp_out_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f" EXPERIMENT: Dataset={dataset_name} | Loss={loss_type}")
    print(f" Target Directory: {exp_out_dir}")
    print(f"=======================================================")

    # 1. Load Data Split & Metadata (Strictly using frozen Phase 0 selected features)
    data = load_dataset_split(
        dataset_name=dataset_name,
        base_data_dir=base_data_dir,
        sample_limit=sample_limit,
        seed=seed,
    )

    # Save Dataset Integration Report
    integration_report = data["integration_report"]
    with open(os.path.join(exp_out_dir, "dataset_integration_report.json"), "w", encoding="utf-8") as f:
        json.dump(integration_report, f, indent=2)

    ds_report_dir = os.path.join(output_dir_base, dataset_name)
    os.makedirs(ds_report_dir, exist_ok=True)
    with open(os.path.join(ds_report_dir, "dataset_integration_report.json"), "w", encoding="utf-8") as f:
        json.dump(integration_report, f, indent=2)

    print(f" [INTEGRATION CONFIRMED] Config: {integration_report['loaded_feature_config_path']}")
    print(f" [INTEGRATION CONFIRMED] Selected Features ({data['input_dim']}): {data['feature_names'][:5]}...")
    print(f" [INTEGRATION CONFIRMED] Num Classes: {data['num_classes']} | Class Names: {data['class_names']}")

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

    # 3. Instantiate Fixed Architecture (Fixed Temporal + Fixed Relational + Concat Fusion + Classifier Head)
    model = MultimodalFusionClassifier(
        fusion_type="concat",
        temp_input_dim=data["input_dim"],
        rel_node_dim=data["input_dim"],
        rel_edge_dim=0,
        num_classes=data["num_classes"],
        dropout_rate=0.2,
    )

    # 4. Instantiate Loss Function
    if loss_type == "standard_ce":
        criterion = StandardCrossEntropyLoss()
    elif loss_type == "weighted_ce":
        criterion = ClassWeightedCrossEntropyLoss(weights=data["class_weights"])
    elif loss_type == "focal_loss":
        criterion = FocalLoss(gamma=2.0)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Save Experiment Config
    config = {
        "dataset": dataset_name,
        "loss": loss_type,
        "gamma": 2.0 if loss_type == "focal_loss" else None,
        "class_weights": data["class_weights"].tolist() if loss_type == "weighted_ce" else None,
        "num_classes": data["num_classes"],
        "class_names": data["class_names"],
        "class_distribution": data["class_distribution"],
        "imbalance_ratio": data["imbalance_ratio"],
        "seed": seed,
        "fusion": "concat",
        "temporal_encoder": "FixedTemporalEncoder (GRU, 2 layers, 64 hidden)",
        "relational_encoder": "FixedRelationalEncoder (Projection + Aggregation)",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": epochs,
        "early_stopping_patience": 5,
    }
    with open(os.path.join(exp_out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 5. Train Model
    checkpoint_path = os.path.join(exp_out_dir, "best_model.pt")
    trainer = ExperimentTrainer(
        model=model,
        criterion=criterion,
        learning_rate=0.001,
    )
    fit_res = trainer.fit(
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        epochs=epochs,
        patience=5,
        save_checkpoint_path=checkpoint_path,
    )

    # 6. Evaluate Held-out Test Set
    test_loss, y_test_true, y_test_pred = trainer.evaluate(data["test_loader"])
    print(f" Test Evaluation Complete: Test Loss = {test_loss:.4f}")

    # 7. Compute Metrics, Plots & Export Results
    metrics = compute_metrics_and_plots(
        y_true=y_test_true,
        y_pred=y_test_pred,
        class_names=data["class_names"],
        output_dir=exp_out_dir,
        exp_prefix=exp_prefix,
        train_history=fit_res,
    )

    print(f" [FINISHED] {exp_prefix} | Test Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Controlled Loss Experiments Harness")
    parser.add_argument("--dataset", type=str, default="all", choices=DATASETS + ["all"], help="Target dataset name")
    parser.add_argument("--loss", type=str, default="all", choices=LOSS_TYPES + ["all"], help="Target loss function")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--sample-limit", type=int, default=100000, help="Max raw records to load per dataset")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw data folder")
    parser.add_argument("--output-dir", type=str, default="results/phase1_loss_experiments", help="Path to results directory")

    args = parser.parse_args()

    target_datasets = DATASETS if args.dataset == "all" else [args.dataset]
    target_losses = LOSS_TYPES if args.loss == "all" else [args.loss]

    total_exps = len(target_datasets) * len(target_losses)
    completed = 0

    print(f"Starting Phase 1 Controlled Experiment Suite ({total_exps} runs planned)...")

    for ds in target_datasets:
        for loss in target_losses:
            completed += 1
            print(f"\nProgress: Run {completed}/{total_exps}")
            try:
                run_single_experiment(
                    dataset_name=ds,
                    loss_type=loss,
                    epochs=args.epochs,
                    sample_limit=args.sample_limit,
                    base_data_dir=args.data_dir,
                    output_dir_base=args.output_dir,
                )
            except Exception as e:
                print(f"[ERROR] Failed experiment {ds} - {loss}: {e}")

    # Generate aggregate Phase 1 summary report if all or multiple completed
    from scripts.generate_phase1_report import generate_phase1_summary
    generate_phase1_summary(results_dir=args.output_dir)


if __name__ == "__main__":
    main()
