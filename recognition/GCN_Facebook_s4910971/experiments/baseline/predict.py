"""Evaluate one trained MLP and save data for later analysis."""

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import load_graph
from experiments.baseline.model import MLP



# Default model when --model is not supplied.
MODEL = ROOT / "training_results" / "mlp_20261029_210341" / "model.pt"
CLASSES = ["company", "government", "politician", "tvshow"]

WARMUP = 20
REPEATS = 100



def main():
    # Select a checkpoint from the run parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    args = parser.parse_args()
    path = ROOT / args.model
    folder = path.parent / "prediction"



    # Restore the training split and model settings from the checkpoint.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    params = ckpt["params"]
    data = load_graph(seed=ckpt["seed"]).to(device)
    model = MLP(data.num_node_features, params["hidden_dim"], 4,
                params["dropout"]).to(device)
    model.load_state_dict(ckpt["weights"])
    model.eval()
    folder.mkdir(parents=True, exist_ok=True)



    # Time whole-graph forward passes after warming up the device.
    with torch.no_grad():
        for _ in range(WARMUP):
            model(data.x)

        if device == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()

        for _ in range(REPEATS):
            out = model(data.x)

        if device == "cuda":
            torch.cuda.synchronize()
        ms = (perf_counter() - start) * 1000 / REPEATS



    # Convert class scores to probabilities for all nodes.
    prob = out.cpu().softmax(dim=1)
    pred = prob.argmax(dim=1)
    y = data.y.cpu()
    test = data.test_mask.cpu()



    # Save test metrics.
    report = classification_report(
        y[test], pred[test], labels=[0, 1, 2, 3],
        target_names=CLASSES, digits=4, zero_division=0,
    )
    auc = roc_auc_score(y[test].numpy(), prob[test].numpy(),
                         labels=[0, 1, 2, 3], multi_class="ovr", average="macro")

    with (folder / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\nMacro AUROC: {auc:.4f}\n")



    # Save node IDs, labels and probabilities.
    columns = ["node", "split", "true", "pred"]
    columns += [f"p_{name}" for name in CLASSES]

    with (folder / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for split, mask in [("val", data.val_mask), ("test", data.test_mask)]:
            nodes = mask.cpu().nonzero(as_tuple=True)[0].tolist()
            for node in nodes:
                writer.writerow([
                    node, split, CLASSES[int(y[node])], CLASSES[int(pred[node])],
                    *prob[node].tolist(),
                ])



    # Extract the last hidden layer. Dropout is off during evaluation.
    with torch.no_grad():
        x = F.relu(model.fc1(data.x))
        x = F.relu(model.fc2(x))

    torch.save({
        "x": x.cpu(),
        "y": y,
        "classes": CLASSES,
        "val_mask": data.val_mask.cpu(),
        "test_mask": test,
    }, folder / "embeddings.pt")



    # Save the model reference and inference time.
    with (folder / "predict_info.txt").open("w", encoding="utf-8") as f:
        f.write(f"Model: {path.parent.name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Inference (ms/graph): {ms:.4f}\n")



if __name__ == "__main__":
    main()
