#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

MUTATIONS = ["F33I", "D58N", "Q72H", "A75V", "V85A", "V154F", "Y180F"]

mapping_path = Path("results/ale_site_scores/test_homolog_mapped_sites.tsv")
out_base = Path("data/mutation_holdout/dgoA")

df = pd.read_csv(mapping_path, sep="\t")

df = df[
    (df["dataset"] == "dgoA") &
    (df["split"] == "test")
].copy()

rows = []

for mut in MUTATIONS:
    sub = df[df["mutation"] == mut].copy()
    out = out_base / f"holdout_{mut}" / "test_heldout_mapping.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, sep="\t", index=False)

    rows.append({
        "heldout_mutation": mut,
        "mapping_rows": len(sub),
        "mapping_file": str(out),
    })

summary = pd.DataFrame(rows)
summary_out = Path("reports/mutation_holdout/dgoa_holdout_mapping_counts.tsv")
summary_out.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(summary_out, sep="\t", index=False)

print(summary)
print()
print(f"Wrote: {summary_out}")
