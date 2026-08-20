#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

genes = ["topA", "yeiB", "spoT", "dgoA"]
tags = ["40", "70", "80", "90", "95"]

rows = []

for gene in genes:
    for tag in tags:
        path = Path(f"results/mmseqs_sweep/{gene}_{tag}_clusters.tsv")

        if not path.exists():
            continue

        df = pd.read_csv(path, sep="\t", header=None, names=["cluster_id", "seq_id"])
        sizes = df.groupby("cluster_id").size().sort_values(ascending=False)

        rows.append({
            "gene": gene,
            "identity": f"{tag}%",
            "sequences": len(df),
            "clusters": len(sizes),
            "largest_cluster": int(sizes.iloc[0]),
            "singleton_clusters": int((sizes == 1).sum()),
        })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

Path("results/mmseqs_sweep").mkdir(parents=True, exist_ok=True)
summary.to_csv("results/mmseqs_sweep/cluster_sweep_summary.tsv", sep="\t", index=False)

print("\nWrote: results/mmseqs_sweep/cluster_sweep_summary.tsv")
