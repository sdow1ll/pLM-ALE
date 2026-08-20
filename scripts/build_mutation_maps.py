#!/usr/bin/env python3
from pathlib import Path
import csv

MIN_IDENTITY = 40.0
MAX_IDENTITY = 100.0
MIN_QCOV = 80.0
MIN_LEN_RATIO = 0.80
MAX_LEN_RATIO = 1.20

MUTATIONS = {
    "topA": [("H33Y", 33, "H", "Y")],
    "yeiB": [("L143I", 143, "L", "I")],
    "spoT": [("K662I", 662, "K", "I")],
    "dgoA": [
        ("F33I", 33, "F", "I"),
        ("D58N", 58, "D", "N"),
        ("Q72H", 72, "Q", "H"),
        ("A75V", 75, "A", "V"),
        ("V85A", 85, "V", "A"),
        ("V154F", 154, "V", "F"),
        ("Y180F", 180, "Y", "F"),
    ],
}

COLUMNS = [
    "sacc", "pident", "qcovs", "length", "qlen", "slen",
    "qstart", "qend", "sstart", "send",
    "evalue", "bitscore", "qseq", "sseq", "stitle"
]

def map_query_position_to_subject(qseq, sseq, qstart, sstart, send, query_pos):
    qpos = int(qstart) - 1

    sstart = int(sstart)
    send = int(send)
    step = 1 if send >= sstart else -1
    spos = sstart - step

    for qaa, saa in zip(qseq, sseq):
        if qaa != "-":
            qpos += 1

        if saa != "-":
            spos += step

        if qaa != "-" and qpos == query_pos:
            if saa == "-":
                return None
            return spos, qaa, saa

    return None

def parse_blast(path):
    rows = []

    with open(path) as f:
        reader = csv.reader(f, delimiter="\t")

        for parts in reader:
            if len(parts) < 14:
                continue

            while len(parts) < len(COLUMNS):
                parts.append("")

            row = dict(zip(COLUMNS, parts[:len(COLUMNS)]))

            try:
                row["pident"] = float(row["pident"])
                row["qcovs"] = float(row["qcovs"])
                row["length"] = int(row["length"])
                row["qlen"] = int(row["qlen"])
                row["slen"] = int(row["slen"])
                row["qstart"] = int(row["qstart"])
                row["qend"] = int(row["qend"])
                row["sstart"] = int(row["sstart"])
                row["send"] = int(row["send"])
                row["bitscore"] = float(row["bitscore"])
            except ValueError:
                continue

            rows.append(row)

    return rows

def main():
    outdir = Path("data/processed/mutation_maps")
    outdir.mkdir(parents=True, exist_ok=True)

    for gene, mutations in MUTATIONS.items():
        blast_path = Path(f"blast_results/{gene}_blast_align.tsv")

        if not blast_path.exists():
            raise FileNotFoundError(blast_path)

        rows = parse_blast(blast_path)
        best = {}

        for row in rows:
            pident = row["pident"]
            qcovs = row["qcovs"]
            qlen = row["qlen"]
            slen = row["slen"]

            if not (MIN_IDENTITY <= pident < MAX_IDENTITY):
                continue

            if qcovs < MIN_QCOV:
                continue

            len_ratio = slen / qlen
            if not (MIN_LEN_RATIO <= len_ratio <= MAX_LEN_RATIO):
                continue

            acc = row["sacc"]

            for mut_name, qpos, wt, alt in mutations:
                mapped = map_query_position_to_subject(
                    row["qseq"],
                    row["sseq"],
                    row["qstart"],
                    row["sstart"],
                    row["send"],
                    qpos,
                )

                if mapped is None:
                    continue

                subject_pos, query_residue, subject_residue = mapped

                if query_residue != wt:
                    continue

                # Critical filter:
                # only keep homologs that actually have the WT residue
                # at the aligned homolog coordinate.
                if subject_residue != wt:
                    continue

                candidate = {
                    "gene": gene,
                    "accession": acc,
                    "mutation": mut_name,
                    "query_pos": qpos,
                    "subject_pos": subject_pos,
                    "wt": wt,
                    "alt": alt,
                    "query_residue": query_residue,
                    "subject_residue": subject_residue,
                    "pident": pident,
                    "qcovs": qcovs,
                    "qlen": qlen,
                    "slen": slen,
                    "bitscore": row["bitscore"],
                    "stitle": row["stitle"],
                }

                key = (acc, mut_name)

                if key not in best or candidate["bitscore"] > best[key]["bitscore"]:
                    best[key] = candidate

        accepted = list(best.values())

        map_path = outdir / f"{gene}_mutation_map.tsv"
        with open(map_path, "w", newline="") as out:
            fieldnames = [
                "gene", "accession", "mutation", "query_pos", "subject_pos",
                "wt", "alt", "query_residue", "subject_residue",
                "pident", "qcovs", "qlen", "slen", "bitscore", "stitle"
            ]
            writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(accepted)

        acc_path = outdir / f"{gene}_eligible_accessions.txt"
        accessions = sorted({row["accession"] for row in accepted})

        with open(acc_path, "w") as out:
            for acc in accessions:
                out.write(acc + "\n")

        print(f"{gene}:")
        print(f"  accepted mutation-mapped rows: {len(accepted)}")
        print(f"  eligible unique accessions:    {len(accessions)}")
        print(f"  wrote: {map_path}")
        print(f"  wrote: {acc_path}")

if __name__ == "__main__":
    main()
