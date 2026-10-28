
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from dataset import load_graph
from experiments.main.model import GCN


DATA_FILE = PROJECT / "data" / "facebook.npz"
RUNS_FOLDER = PROJECT / "results" / "main"


def macro_f1(actual, predicted):
    confusion = torch.bincount(actual * 4 + predicted, minlength=16).reshape(4, 4)
    tp = confusion.diag().float()
    precision = tp / confusion.sum(dim=0).clamp(min=1)
    recall = tp / confusion.sum(dim=1).clamp(min=1)
    return (2 * precision * recall / (precision + recall).clamp(min=1e-12)).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--runs", type=Path, default=RUNS_FOLDER)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph = load_graph(args.data, seed=42).to(device)
    model = GCN(graph.num_node_features, 32, 4, 0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=0.0)
    output = args.runs / f"main_train_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    history = []

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        scores = model(graph.x, graph.edge_index)
        train_loss = F.cross_entropy(scores[graph.train_mask], graph.y[graph.train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            scores = model(graph.x, graph.edge_index)[graph.val_mask]
            actual = graph.y[graph.val_mask]
            predicted = scores.argmax(dim=1)
            val_loss = F.cross_entropy(scores, actual).item()
            val_accuracy = (predicted == actual).float().mean().item()
            val_f1 = macro_f1(actual, predicted)
        history.append([epoch, train_loss.item(), val_loss, val_accuracy, val_f1])

    if device == "cuda":
        torch.cuda.synchronize()
        peak_memory_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_memory_mib = None
    training_seconds = time.perf_counter() - start

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"])
        writer.writerows(history)
    torch.save({"model": "main", "weights": model.state_dict(), "depth": 4,
                "hidden_dim": 32, "dropout": 0.2, "learning_rate": 0.02,
                "weight_decay": 0.0, "epoch": args.epochs, "seed": 42}, output / "last.pt")
    (output / "validation_result.json").write_text(json.dumps({
        "val_accuracy": val_accuracy, "val_macro_f1": val_f1,
        "training_seconds": training_seconds, "peak_gpu_memory_mib": peak_memory_mib,
        "parameters": sum(p.numel() for p in model.parameters()),
    }, indent=2), encoding="utf-8")
    print(f"Validation accuracy: {val_accuracy:.2%}, macro F1: {val_f1:.2%}")
    print(f"Training results: {output}")


if __name__ == "__main__":
    main()
