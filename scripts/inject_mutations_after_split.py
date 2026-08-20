#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

def read_fasta(path):
    records = {}
    header = None
    seq_lines = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(seq_lines).upper()
                header = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            records[header] = "".join(seq_lines).upper()

    return records

def write_record(out, header, seq):
    out.write(f">{header}\n")
    for i in range(0, len(seq), 80):
        out.write(seq[i:i+80] + "\n")

parser = argparse.ArgumentParser()
parser.add_argument("--gene", required=True)
parser.add_argument("--fasta", required=True)
parser.add_argument("--mutation_map", required=True)
parser.add_argument("--split_table", required=True)
parser.add_argument("--outdir", required=True)
args = parser.parse_args()

records = read_fasta(args.fasta)
mutation_map = pd.read_csv(args.mutation_map, sep="\t")
split_table = pd.read_csv(args.split_table, sep="\t")

seq_to_split = dict(zip(split_table["seq_id"], split_table["split"]))
seq_to_cluster = dict(zip(split_table["seq_id"], split_table["cluster_id"]))

Path(args.outdir).mkdir(parents=True, exist_ok=True)

out_handles = {
    split: open(Path(args.outdir) / f"{split}_mutated.faa", "w")
    for split in ["train", "val", "test"]
}

stats = {"train": 0, "val": 0, "test": 0, "skipped": 0}

for _, row in mutation_map.iterrows():
    acc = row["accession"]

    if acc not in records:
        stats["skipped"] += 1
        continue

    if acc not in seq_to_split:
        stats["skipped"] += 1
        continue

    split = seq_to_split[acc]
    cluster_id = seq_to_cluster[acc]

    seq = records[acc]
    subject_pos = int(row["subject_pos"])
    idx = subject_pos - 1

    wt = row["wt"]
    alt = row["alt"]
    mutation = row["mutation"]

    if idx < 0 or idx >= len(seq):
        stats["skipped"] += 1
        continue

    if seq[idx] != wt:
        stats["skipped"] += 1
        continue

    mutated = seq[:idx] + alt + seq[idx+1:]

    header = (
        f"{acc}|gene={args.gene}|mutation={mutation}|"
        f"mapped_pos={subject_pos}|cluster={cluster_id}|split={split}"
    )

    write_record(out_handles[split], header, mutated)
    stats[split] += 1

for handle in out_handles.values():
    handle.close()

print(f"Gene: {args.gene}")
for k, v in stats.items():
    print(f"  {k}: {v}")
