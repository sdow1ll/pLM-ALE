#!/usr/bin/env python3
import argparse
from pathlib import Path

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
                    records.append((header, "".join(seq_lines).upper()))
                header = line[1:].split()[0]
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            records.append((header, "".join(seq_lines).upper()))

    return records

def write_fasta(records, path):
    with open(path, "w") as out:
        for header, seq in records:
            out.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + "\n")

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

records = read_fasta(args.input)

seen = set()
deduped = []

for header, seq in records:
    if seq in seen:
        continue
    seen.add(seq)
    deduped.append((header, seq))

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
write_fasta(deduped, args.output)

print(f"Input records:  {len(records)}")
print(f"Output records: {len(deduped)}")
print(f"Wrote: {args.output}")
