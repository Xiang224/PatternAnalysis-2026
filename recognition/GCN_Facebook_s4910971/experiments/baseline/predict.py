"""Load an MLP checkpoint and predict every test node."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from dataset import load_graph
from experiments.baseline.model import MLP


DATA_FILE = PROJECT / "data" / "facebook.npz"
RUNS_FOLDER = PROJECT / "results" / "baseline"
DEFAULT_WEIGHTS = RUNS_FOLDER / "mlp3_gridsearch_20261022_120742" / "best.pt"
CLASS_NAMES = ["company", "government", "politician", "tvshow"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--runs", type=Path, default=RUNS_FOLDER)
    args = parser.parse_args()

    graph = load_graph(args.data, seed=42)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = checkpoint["weights"] if "weights" in checkpoint else checkpoint
    model = MLP(graph.num_node_features, weights["fc1.weight"].shape[0], 4)
    model.load_state_dict(weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    test_ids = graph.test_mask.nonzero(as_tuple=True)[0]
    actual = graph.y[test_ids].numpy()
    with torch.no_grad():
        scores = model(graph.x[test_ids].to(device))
        probabilities = scores.softmax(dim=1).cpu().numpy()
    predicted = probabilities.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=range(4), zero_division=0
    )
    metrics = {
        "checkpoint": str(args.checkpoint), "test_nodes": len(actual),
        "accuracy": float((predicted == actual).mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "macro_auroc_ovr": float(roc_auc_score(actual, probabilities, multi_class="ovr", average="macro")),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=range(4)).tolist(),
        "classes": {name: {"support": int(support[i]), "precision": float(precision[i]),
                            "recall": float(recall[i]), "f1": float(f1[i])}
                    for i, name in enumerate(CLASS_NAMES)},
    }

    output = args.runs / f"baseline_prediction_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["node_id", "actual_class", "predicted_class", "confidence", "correct"])
        for node_id, truth, guess, prob in zip(test_ids.tolist(), actual, predicted, probabilities):
            writer.writerow([node_id, CLASS_NAMES[int(truth)], CLASS_NAMES[int(guess)],
                             float(prob.max()), bool(truth == guess)])
    (output / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"MLP test accuracy: {metrics['accuracy']:.2%}, macro F1: {metrics['macro_f1']:.2%}")
    print(f"Prediction results: {output}")


if __name__ == "__main__":
    main()
