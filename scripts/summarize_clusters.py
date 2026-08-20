#!/usr/bin/env python3
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--clusters", required=True)
args = parser.parse_args()

df = pd.read_csv(args.clusters, sep="\t", header=None, names=["cluster_id", "seq_id"])
sizes = df.groupby("cluster_id").size().sort_values(ascending=False)

print(f"File: {args.clusters}")
print(f"Sequences: {len(df)}")
print(f"Clusters: {len(sizes)}")
print(f"Largest cluster: {sizes.iloc[0]}")
print(f"Singleton clusters: {(sizes == 1).sum()}")
print()
print("Top 10 cluster sizes:")
print(sizes.head(10).to_string())
