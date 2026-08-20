#!/usr/bin/env python3
from pathlib import Path
import re
import pandas as pd

MUTATIONS = ["F33I", "D58N", "Q72H", "A75V", "V85A", "V154F", "Y180F"]

BASE = Path("data/final_cluster95/dgoA")
OUT_BASE = Path("data/mutation_holdout/dgoA")
REPORT = Path("reports/mutation_holdout/dgoa_leave_one_mutation_out_counts.tsv")

def read_fasta(path):
    records = []
    header = None
    seq_lines = []

    with open(path) as f:
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

    return records

def write_fasta(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as out:
        for header, seq in records:
            out.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + "\n")

def get_mutation(header):
    m = re.search(r"(?:^|\|)mutation=([^|]+)", header)
    if m is None:
        raise ValueError(f"Could not find mutation=... in header: {header}")
    return m.group(1)

def count_by_mutation(records):
    counts = {}
    for header, _ in records:
        mut = get_mutation(header)
        counts[mut] = counts.get(mut, 0) + 1
    return counts

def main():
    train_records = read_fasta(BASE / "train.faa")
    val_records = read_fasta(BASE / "val.faa")
    test_records = read_fasta(BASE / "test.faa")

    rows = []

    for heldout in MUTATIONS:
        fold_dir = OUT_BASE / f"holdout_{heldout}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_keep = [(h, s) for h, s in train_records if get_mutation(h) != heldout]
        val_keep = [(h, s) for h, s in val_records if get_mutation(h) != heldout]

        # Primary mutation-held-out test set:
        # held-out mutation only, from the original cluster95 test split.
        test_heldout = [(h, s) for h, s in test_records if get_mutation(h) == heldout]

        # Optional: seen-mutation test set for comparison.
        test_seen = [(h, s) for h, s in test_records if get_mutation(h) != heldout]

        write_fasta(train_keep, fold_dir / "train.faa")
        write_fasta(val_keep, fold_dir / "val.faa")
        write_fasta(test_heldout, fold_dir / "test_heldout_mutation.faa")
        write_fasta(test_seen, fold_dir / "test_seen_mutations.faa")

        rows.append({
            "heldout_mutation": heldout,
            "train_sequences": len(train_keep),
            "val_sequences": len(val_keep),
            "test_heldout_sequences": len(test_heldout),
            "test_seen_sequences": len(test_seen),
        })

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(REPORT, sep="\t", index=False)

    print(df)
    print()
    print(f"Wrote: {REPORT}")
    print(f"Wrote fold FASTAs under: {OUT_BASE}")

if __name__ == "__main__":
    main()
