#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--clusters", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_frac", type=float, default=0.80)
parser.add_argument("--val_frac", type=float, default=0.10)
args = parser.parse_args()

df = pd.read_csv(args.clusters, sep="\t", header=None, names=["cluster_id", "seq_id"])

cluster_sizes = (
    df.groupby("cluster_id")
    .size()
    .reset_index(name="size")
    .sample(frac=1.0, random_state=args.seed)
    .sort_values("size", ascending=False)
    .reset_index(drop=True)
)

total = int(cluster_sizes["size"].sum())

targets = {
    "train": args.train_frac * total,
    "val": args.val_frac * total,
    "test": (1.0 - args.train_frac - args.val_frac) * total,
}

counts = {"train": 0, "val": 0, "test": 0}
cluster_to_split = {}

for _, row in cluster_sizes.iterrows():
    cid = row["cluster_id"]
    size = int(row["size"])

    # Assign each cluster to the split that is currently most below target.
    deficits = {
        split: targets[split] - counts[split]
        for split in ["train", "val", "test"]
    }

    split = max(deficits, key=deficits.get)
    cluster_to_split[cid] = split
    counts[split] += size

df["split"] = df["cluster_id"].map(cluster_to_split)
df = df[["seq_id", "cluster_id", "split"]]

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out, sep="\t", index=False)

print(f"Wrote: {args.out}")
print()
print("Sequence counts:")
print(df["split"].value_counts().to_string())
print()
print("Sequence percentages:")
print((100 * df["split"].value_counts(normalize=True)).round(2).to_string())
print()
print("Cluster counts:")
cluster_split_df = pd.DataFrame(
    [{"cluster_id": c, "split": s} for c, s in cluster_to_split.items()]
)
print(cluster_split_df["split"].value_counts().to_string())
