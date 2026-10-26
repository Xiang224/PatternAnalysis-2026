
"""Search 27 MLP configurations using validation scores only."""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from dataset import load_graph
from experiments.baseline.model import MLP


DATA_FILE = PROJECT / "data" / "facebook.npz"
RUNS_FOLDER = PROJECT / "results" / "baseline"
PARAMETERS = {
    "hidden_dim": [16, 32, 64],
    "learning_rate": [0.005, 0.01, 0.02],
    "dropout": [0.2, 0.5, 0.7],
}
EPOCHS = 200
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--runs", type=Path, default=RUNS_FOLDER)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph = load_graph(args.data, seed=SEED).to(device)
    run_dir = args.runs / f"mlp3_gridsearch_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)

    settings = {"layers": 3, "parameters": PARAMETERS, "epochs": args.epochs, "seed": SEED,
                "split": "70/10/20", "selection": f"epoch-{args.epochs} validation macro F1",
                "weight_decay": 0.0005, "device": device}
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    fields = ["trial", "hidden_dim", "learning_rate", "dropout", "epochs",
              "val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1",
              "val_loss", "training_seconds", "peak_gpu_memory_mib", "parameters"]
    best = None

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as summary_file, \
         (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as history_file:
        summary = csv.DictWriter(summary_file, fieldnames=fields)
        history = csv.writer(history_file)
        summary.writeheader()
        history.writerow(["trial", "epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"])

        for trial, params in enumerate(ParameterGrid(PARAMETERS), start=1):
            torch.manual_seed(SEED)
            model = MLP(graph.num_node_features, params["hidden_dim"], 4, params["dropout"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=0.0005)

            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()

            for epoch in range(1, args.epochs + 1):
                model.train()
                optimizer.zero_grad()
                scores = model(graph.x)
                train_loss = F.cross_entropy(scores[graph.train_mask], graph.y[graph.train_mask])
                train_loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    scores = model(graph.x)[graph.val_mask]
                    labels = graph.y[graph.val_mask]
                    predictions = scores.argmax(dim=1)
                    val_loss = F.cross_entropy(scores, labels).item()
                    val_accuracy = (predictions == labels).float().mean().item()
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        labels.cpu(), predictions.cpu(), labels=[0, 1, 2, 3], zero_division=0
                    )
                    val_macro_precision = float(precision.mean())
                    val_macro_recall = float(recall.mean())
                    val_macro_f1 = float(f1.mean())

                history.writerow([trial, epoch, train_loss.item(), val_loss, val_accuracy, val_macro_f1])

            if device == "cuda":
                torch.cuda.synchronize()
                memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                memory = None

            result = {
                "trial": trial, **params, "epochs": args.epochs,
                "val_accuracy": val_accuracy,
                "val_macro_precision": val_macro_precision,
                "val_macro_recall": val_macro_recall,
                "val_macro_f1": val_macro_f1,
                "val_loss": val_loss,
                "training_seconds": time.perf_counter() - start,
                "peak_gpu_memory_mib": memory,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            summary.writerow(result)
            summary_file.flush()
            history_file.flush()

            if best is None or (result["val_macro_f1"], -result["val_loss"]) > (best["val_macro_f1"], -best["val_loss"]):
                best = result
                torch.save(model.state_dict(), run_dir / "best.pt")
                (run_dir / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    print(f"Best configuration: hidden={best['hidden_dim']}, lr={best['learning_rate']}, "
          f"dropout={best['dropout']}; validation macro F1={best['val_macro_f1']:.4f}")
    print(f"Results folder: {run_dir}")


if __name__ == "__main__":
    main()
