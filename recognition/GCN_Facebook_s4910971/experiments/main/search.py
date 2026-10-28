
import csv
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from dataset import load_graph
from experiments.main.model import GCN


DATA_FILE = PROJECT / "data" / "facebook.npz"
RUNS_FOLDER = PROJECT / "results" / "main"
CLASS_NAMES = ["company", "government", "politician", "tvshow"]

SEED = 42
DEPTHS = [2, 3, 4, 6, 8]
HIDDEN_DIMS = [16, 32, 64]
LEARNING_RATES = [0.005, 0.01, 0.02]
DROPOUTS = [0.2, 0.5, 0.7]
MAX_EPOCHS = 200
WEIGHT_DECAYS = [0.0, 0.0005, 0.005]


def train_one(graph, device, run_folder, depth, hidden_dim, learning_rate, dropout, weight_decay):
    torch.manual_seed(SEED)
    model = GCN(graph.num_node_features, hidden_dim, depth, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    trial_folder = run_folder / f"depth{depth}_width{hidden_dim}_lr{learning_rate}_dropout{dropout}_wd{weight_decay}"
    trial_folder.mkdir()
    checkpoint_file = trial_folder / "last.pt"
    history = []

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        scores = model(graph.x, graph.edge_index)
        train_loss = F.cross_entropy(scores[graph.train_mask], graph.y[graph.train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = model(graph.x, graph.edge_index)
            val_scores = scores[graph.val_mask]
            val_labels = graph.y[graph.val_mask]
            val_loss = F.cross_entropy(val_scores, val_labels).item()
            val_predictions = val_scores.argmax(dim=1)
            val_accuracy = (val_predictions == val_labels).float().mean().item()
            class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
                val_labels.cpu(), val_predictions.cpu(), labels=range(len(CLASS_NAMES)),
                zero_division=0,
            )
            val_precision = class_precision.mean()
            val_recall = class_recall.mean()
            val_f1 = class_f1.mean()

        history.append([
            epoch, train_loss.item(), val_loss, val_accuracy,
            val_precision, val_recall, val_f1,
        ])

    if device == "cuda":
        torch.cuda.synchronize()
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_memory_mb = None
    training_seconds = time.perf_counter() - start
    torch.save({
        "weights": model.state_dict(), "depth": depth,
        "hidden_dim": hidden_dim, "dropout": dropout,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "epoch": MAX_EPOCHS,
    }, checkpoint_file)

    with (trial_folder / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_accuracy",
            "val_macro_precision", "val_macro_recall", "val_macro_f1",
        ])
        writer.writerows(history)

    result = {
        "depth": depth, "hidden_dim": hidden_dim,
        "learning_rate": learning_rate, "dropout": dropout,
        "weight_decay": weight_decay,
        "epochs_run": len(history),
        "val_accuracy": val_accuracy,
        "val_macro_precision": val_precision,
        "val_macro_recall": val_recall,
        "val_macro_f1": val_f1,
        "val_loss": val_loss, "training_seconds": training_seconds,
        "peak_gpu_memory_mb": peak_memory_mb,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(checkpoint_file),
    }
    for index, name in enumerate(CLASS_NAMES):
        result[f"val_precision_{name}"] = class_precision[index]
        result[f"val_recall_{name}"] = class_recall[index]
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph = load_graph(DATA_FILE, seed=SEED).to(device)
    run_folder = RUNS_FOLDER / f"gcn_search_{datetime.now():%Y%m%d_%H%M%S}"
    run_folder.mkdir(parents=True, exist_ok=False)

    (run_folder / "settings.json").write_text(json.dumps({
        "seed": SEED, "split": "70/10/20", "data_file": str(DATA_FILE),
        "depths": DEPTHS, "hidden_dims": HIDDEN_DIMS,
        "learning_rates": LEARNING_RATES, "dropouts": DROPOUTS,
        "weight_decays": WEIGHT_DECAYS,
        "epochs": MAX_EPOCHS,
        "optimizer": "Adam",
        "checkpoint": "last epoch of every combination",
        "device": device,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = [
        "depth", "hidden_dim", "learning_rate", "dropout", "weight_decay", "epochs_run",
        "val_accuracy", "val_macro_precision", "val_macro_recall",
        "val_macro_f1", "val_loss",
        "training_seconds", "peak_gpu_memory_mb", "parameters", "checkpoint",
    ]
    for name in CLASS_NAMES:
        fields.extend([f"val_precision_{name}", f"val_recall_{name}"])
    with (run_folder / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for values in product(DEPTHS, HIDDEN_DIMS, LEARNING_RATES, DROPOUTS, WEIGHT_DECAYS):
            result = train_one(graph, device, run_folder, *values)
            writer.writerow(result)
            file.flush()
    print(f"Trial results saved to: {run_folder / 'summary.csv'}")


if __name__ == "__main__":
    main()
