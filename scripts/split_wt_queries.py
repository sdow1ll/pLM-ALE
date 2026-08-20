#!/usr/bin/env python3
from pathlib import Path

input_fasta = Path("data/raw/wt_queries.faa")
outdir = Path("data/raw/queries/by_gene")
outdir.mkdir(parents=True, exist_ok=True)

records = []
header = None
seq_lines = []

with open(input_fasta) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_lines)))
            header = line[1:]
            seq_lines = []
        else:
            seq_lines.append(line)

    if header is not None:
        records.append((header, "".join(seq_lines)))

for header, seq in records:
    gene = header.replace("_WT", "")

    outpath = outdir / f"{gene}_query.faa"
    with open(outpath, "w") as out:
        out.write(f">{gene}_WT\n")
        for i in range(0, len(seq), 80):
            out.write(seq[i:i+80] + "\n")

    print(f"Wrote {outpath} length={len(seq)}")
