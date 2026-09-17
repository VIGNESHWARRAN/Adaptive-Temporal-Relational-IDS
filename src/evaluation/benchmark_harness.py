"""
Benchmark Harness for Comparative Multimodal Fusion Evaluation.
Evaluates all 6 fusion strategies (concat, gated, bilinear, cross_attention, decision_ensemble, quantum_pqc)
under identical frozen encoders across Experiments E1-E5.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.classifiers import MultimodalFusionClassifier


class SampleDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        seqs = torch.stack([b["sequence_S_t"] for b in batch], dim=0)
        labels = torch.cat([b["label_binary"] for b in batch], dim=0)
        graphs = [b["graph_G_t"] for b in batch]
        families = [b["label_attack_family"] for b in batch]
        return {"sequences": seqs, "graphs": graphs, "labels": labels, "attack_families": families}


class BenchmarkEvaluator:
    def __init__(self, embed_dim: int = 64, num_classes: int = 2, device: str = "cpu"):
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fusion_methods = [
            "concat",
            "gated",
            "bilinear",
            "cross_attention",
            "decision_ensemble",
            "quantum_pqc",
        ]

    def _train_single_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        epochs: int = 3,
        lr: float = 1e-3,
    ) -> float:
        model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        final_loss = 0.0
        for ep in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                seqs = batch["sequences"].to(self.device)
                labels = batch["labels"].to(self.device)
                graphs = batch["graphs"]

                optimizer.zero_grad()
                logits = model(seqs, graphs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            final_loss = total_loss / max(len(train_loader), 1)

        return final_loss

    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        model.to(self.device)
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                seqs = batch["sequences"].to(self.device)
                labels = batch["labels"].to(self.device)
                graphs = batch["graphs"]

                logits = model(seqs, graphs)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())

        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)

        tp = np.sum((preds_arr == 1) & (targets_arr == 1))
        fp = np.sum((preds_arr == 1) & (targets_arr == 0))
        tn = np.sum((preds_arr == 0) & (targets_arr == 0))
        fn = np.sum((preds_arr == 0) & (targets_arr == 1))

        acc = float((tp + tn) / max(len(targets_arr), 1))
        precision = float(tp / max(tp + fp, 1e-5))
        recall = float(tp / max(tp + fn, 1e-5))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-5))
        fpr = float(fp / max(fp + tn, 1e-5))

        return {
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
        }

    def run_fusion_benchmark(
        self,
        samples: List[Dict[str, Any]],
        train_split: float = 0.7,
        epochs: int = 3,
        batch_size: int = 16,
    ) -> Dict[str, Dict[str, float]]:
        """
        Runs training & evaluation for all 6 fusion strategies on the provided samples.
        """
        n_train = int(len(samples) * train_split)
        train_samples = samples[:n_train]
        test_samples = samples[n_train:]

        train_ds = SampleDataset(train_samples)
        test_ds = SampleDataset(test_samples)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

        results = {}
        print(f"=== Running Controlled 6-Method Fusion Benchmark ({len(train_samples)} Train, {len(test_samples)} Test) ===")

        for fusion_name in self.fusion_methods:
            print(f"--> Training Fusion Method: {fusion_name.upper()}...")
            model = MultimodalFusionClassifier(
                fusion_type=fusion_name,
                embed_dim=self.embed_dim,
                num_classes=self.num_classes,
            )

            loss = self._train_single_model(model, train_loader, epochs=epochs)
            metrics = self._evaluate_model(model, test_loader)
            metrics["final_train_loss"] = round(loss, 4)
            results[fusion_name] = metrics
            print(f"    Results for {fusion_name}: F1={metrics['f1_score']}, Acc={metrics['accuracy']}, Recall={metrics['recall']}, FPR={metrics['false_positive_rate']}")

        return results
