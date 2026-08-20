#!/usr/bin/env python3
import re
from pathlib import Path
import pandas as pd


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
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            records.append((header, "".join(seq_lines).upper()))

    return records


def parse_header(header):
    parts = header.split("|")
    accession = parts[0]

    meta = {}
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            meta[k] = v

    return accession, meta


def parse_mutation(mutation):
    """
    Examples:
      H33Y
      L143I
      V154F
    """
    m = re.match(r"^([A-Z])([0-9]+)([A-Z])$", mutation)
    if m is None:
        raise ValueError(f"Could not parse mutation: {mutation}")

    wt = m.group(1)
    ref_pos = int(m.group(2))
    alt = m.group(3)

    return wt, ref_pos, alt


def main():
    datasets = {
        "ecoli": Path("data/final_cluster95/ecoli/test.faa"),
        "dgoA": Path("data/final_cluster95/dgoA/test.faa"),
    }

    rows = []

    for dataset, fasta_path in datasets.items():
        if not fasta_path.exists():
            raise FileNotFoundError(fasta_path)

        for header, seq in read_fasta(fasta_path):
            accession, meta = parse_header(header)

            gene = meta.get("gene")
            mutation = meta.get("mutation")
            mapped_pos = int(meta.get("mapped_pos"))
            cluster = meta.get("cluster")
            split = meta.get("split")

            wt, reference_pos, alt = parse_mutation(mutation)

            idx = mapped_pos - 1
            residue_at_mapped_pos = seq[idx] if 0 <= idx < len(seq) else None

            rows.append({
                "dataset": dataset,
                "split": split,
                "gene": gene,
                "accession": accession,
                "mutation": mutation,
                "wt": wt,
                "reference_pos_1based": reference_pos,
                "alt": alt,
                "homolog_mapped_pos_1based": mapped_pos,
                "residue_at_homolog_mapped_pos": residue_at_mapped_pos,
                "expected_mutant_residue": alt,
                "mapped_pos_matches_alt": residue_at_mapped_pos == alt,
                "cluster": cluster,
                "sequence_length": len(seq),
                "fasta_header": header,
            })

    df = pd.DataFrame(rows)

    out = Path("results/ale_site_scores/test_homolog_mapped_sites.tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)

    print(df[[
        "dataset",
        "gene",
        "mutation",
        "accession",
        "reference_pos_1based",
        "homolog_mapped_pos_1based",
        "residue_at_homolog_mapped_pos",
        "mapped_pos_matches_alt",
        "sequence_length",
    ]].head(20))

    print()
    print("Summary:")
    print(df.groupby(["dataset", "gene", "mutation"]).size().reset_index(name="test_sequences"))

    print()
    print("Mapped position check:")
    print(df.groupby(["dataset", "mapped_pos_matches_alt"]).size().reset_index(name="count"))

    print()
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
