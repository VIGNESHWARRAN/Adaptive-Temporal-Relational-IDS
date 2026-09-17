"""Experiment Trainer and Evaluator for Phase 1 Controlled Experiments.

Handles PyTorch model training, early stopping, validation tracking, evaluation metrics,
confusion matrix plotting, loss curve generation, and results export.
"""

import time
import os
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ExperimentTrainer:
    """Standardized PyTorch Trainer for Phase 1 Loss Experiments."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = None,
    ):
        self.model = model
        self.criterion = criterion
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Trains for one epoch and returns average loss."""
        self.model.train()
        total_loss = 0.0
        samples_count = 0

        for batch_temp, batch_rel_node, batch_y in train_loader:
            batch_temp = batch_temp.to(self.device)
            batch_rel_node = batch_rel_node.to(self.device)
            batch_y = batch_y.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            logits = self.model(batch_temp, batch_rel_node)
            loss = self.criterion(logits, batch_y)
            loss.backward()
            self.optimizer.step()

            batch_size = batch_y.size(0)
            total_loss += loss.item() * batch_size
            samples_count += batch_size

        return total_loss / max(1, samples_count)

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluates model on data_loader, returning (average loss, y_true, y_pred)."""
        self.model.eval()
        total_loss = 0.0
        samples_count = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_temp, batch_rel_node, batch_y in data_loader:
                batch_temp = batch_temp.to(self.device)
                batch_rel_node = batch_rel_node.to(self.device)
                batch_y_dev = batch_y.to(self.device, dtype=torch.long)

                logits = self.model(batch_temp, batch_rel_node)
                loss = self.criterion(logits, batch_y_dev)

                batch_size = batch_y.size(0)
                total_loss += loss.item() * batch_size
                samples_count += batch_size

                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy())

        avg_loss = total_loss / max(1, samples_count)
        return avg_loss, np.array(all_targets), np.array(all_preds)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        patience: int = 5,
        save_checkpoint_path: str = None,
    ) -> Dict[str, Any]:
        """Trains model with early stopping on validation loss."""
        train_losses = []
        val_losses = []
        val_macro_f1s = []

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, y_val_true, y_val_pred = self.evaluate(val_loader)
            
            _, _, val_f1, _ = precision_recall_fscore_support(
                y_val_true, y_val_pred, average="macro", zero_division=0
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_macro_f1s.append(float(val_f1))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                if save_checkpoint_path:
                    os.makedirs(os.path.dirname(save_checkpoint_path), exist_ok=True)
                    torch.save(self.model.state_dict(), save_checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        training_time = time.time() - start_time

        # Load best checkpoint if saved
        if save_checkpoint_path and os.path.exists(save_checkpoint_path):
            self.model.load_state_dict(torch.load(save_checkpoint_path, map_location=self.device))

        return {
            "training_time": training_time,
            "epochs_completed": len(train_losses),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_val_loss": val_losses[-1],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_macro_f1s": val_macro_f1s,
        }


def compute_metrics_and_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: str,
    exp_prefix: str,
    train_history: Dict[str, Any],
) -> Dict[str, Any]:
    """Calculates all mandatory evaluation metrics, confusion matrix, loss curves, and saves artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Macro & Weighted Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # 2. Per-class Metrics
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class_report = {}
    for idx, cname in enumerate(class_names):
        per_class_report[str(cname)] = {
            "precision": float(per_p[idx]),
            "recall": float(per_r[idx]),
            "f1_score": float(per_f1[idx]),
            "support": int(per_sup[idx]),
        }

    # 3. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(output_dir, f"{exp_prefix}_cm.png")
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {exp_prefix}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 4. Loss Curve Plot
    curve_path = os.path.join(output_dir, f"{exp_prefix}_loss_curve.png")
    epochs = range(1, len(train_history["train_losses"]) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_history["train_losses"], label="Train Loss", marker="o")
    plt.plot(epochs, train_history["val_losses"], label="Validation Loss", marker="s")
    plt.axvline(x=train_history["best_epoch"], color="r", linestyle="--", label=f"Best Epoch ({train_history['best_epoch']})")
    plt.title(f"Training & Validation Loss Curve: {exp_prefix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=300)
    plt.close()

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(w_p),
        "weighted_recall": float(w_r),
        "weighted_f1": float(w_f1),
        "training_time_seconds": float(train_history["training_time"]),
        "epochs_completed": train_history["epochs_completed"],
        "best_epoch": train_history["best_epoch"],
        "best_val_loss": float(train_history["best_val_loss"]),
        "final_val_loss": float(train_history["final_val_loss"]),
        "per_class_metrics": per_class_report,
        "confusion_matrix": cm.tolist(),
        "cm_plot_path": cm_path,
        "loss_curve_path": curve_path,
    }

    # Save metrics JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
