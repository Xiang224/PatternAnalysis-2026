"""Search GCN parameters and save validation results."""

import csv
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


# Import the model and data loader from the project folder.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset import load_graph
from modules import GCN



# Data and output folders.
DATA = ROOT / "data" / "facebook.npz"
RUNS = ROOT / "results" / "main"


# Fixed training settings and search ranges.
SEED = 42
EPOCHS = 200
DEPTHS = [2, 3, 4, 6, 8]
WIDTHS = [16, 32, 64]
LRS = [0.005, 0.01, 0.02]
DROPS = [0.2, 0.5, 0.7]
DECAYS = [0.0, 0.0005, 0.005]



def train_one(data, device, depth, width, lr, drop, wd):
    # Create a model with the same seed for each combination.
    torch.manual_seed(SEED)
    model = GCN(data.num_node_features, width, depth, drop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


    # Train using training-node labels only.
    model.train()
    for epoch in range(EPOCHS):
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()


    # Evaluate the final epoch on validation nodes.
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)[data.val_mask]
        y = data.y[data.val_mask]
        pred = out.argmax(dim=1)
        loss = F.cross_entropy(out, y).item()
        acc = (pred == y).float().mean().item()
        f1 = f1_score(y.cpu(), pred.cpu(), labels=[0, 1, 2, 3],
                      average="macro", zero_division=0)

    return {
        "depth": depth,
        "hidden_dim": width,
        "learning_rate": lr,
        "dropout": drop,
        "weight_decay": wd,
        "val_accuracy": acc,
        "val_macro_f1": f1,
        "val_loss": loss,
    }



def main():
    # Load the fixed graph and create a new output folder.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_graph(DATA, seed=SEED).to(device)
    run = RUNS / f"gcn_search_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True)

    columns = [
        "depth", "hidden_dim", "learning_rate", "dropout", "weight_decay",
        "val_accuracy", "val_macro_f1", "val_loss",
    ]
    best = None


    # Train all combinations and save one row for each.
    with (run / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for depth, width, lr, drop, wd in product(DEPTHS, WIDTHS, LRS, DROPS, DECAYS):
            row = train_one(data, device, depth, width, lr, drop, wd)
            writer.writerow(row)
            f.flush()

            # Prefer higher macro F1, then lower loss if F1 is equal.
            if best is None:
                best = row
            elif row["val_macro_f1"] > best["val_macro_f1"]:
                best = row
            elif row["val_macro_f1"] == best["val_macro_f1"]:
                if row["val_loss"] < best["val_loss"]:
                    best = row


    # Save readable parameters for manual use in train.py.
    with (run / "best_params.txt").open("w", encoding="utf-8") as f:
        f.write("Selection: highest final-epoch validation macro F1; ties use lower validation loss.\n")
        f.write(f"Epochs per combination: {EPOCHS}\nSeed: {SEED}\n")
        f.write("Split: 70% training / 10% validation / 20% test\nOptimizer: Adam\n")
        f.write(f"Depths searched: {DEPTHS}\nHidden widths searched: {WIDTHS}\n")
        f.write(f"Learning rates searched: {LRS}\nDropouts searched: {DROPS}\n")
        f.write(f"Weight decays searched: {DECAYS}\n\n")
        for name, value in best.items():
            f.write(f"{name} = {value}\n")

    print(f"Best: depth={best['depth']}, width={best['hidden_dim']}, "
          f"lr={best['learning_rate']}, dropout={best['dropout']}, wd={best['weight_decay']}")
    print(f"Saved: {run}")



if __name__ == "__main__":
    main()
