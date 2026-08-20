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

def clean_subject_sequence(sseq):
    return sseq.replace("-", "").upper()

def map_query_position_to_extracted_subject_position(qseq, sseq, qstart, query_pos):
    """
    Returns:
      extracted_subject_pos, query_residue, subject_residue

    extracted_subject_pos is 1-based coordinate in the ungapped extracted sseq.
    This is what we will use for mutation injection later.
    """
    qpos = int(qstart) - 1
    extracted_subject_pos = 0

    for qaa, saa in zip(qseq, sseq):
        if qaa != "-":
            qpos += 1

        if saa != "-":
            extracted_subject_pos += 1

        if qaa != "-" and qpos == query_pos:
            if saa == "-":
                return None
            return extracted_subject_pos, qaa, saa

    return None

def write_fasta(records, outpath):
    with open(outpath, "w") as out:
        for header, seq in records:
            out.write(f">{header}\n")
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + "\n")

def main():
    fasta_dir = Path("data/processed/blast_derived_homologs")
    map_dir = Path("data/processed/blast_derived_mutation_maps")
    fasta_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    for gene, mutations in MUTATIONS.items():
        blast_path = Path(f"blast_results/{gene}_blast_align.tsv")
        rows = parse_blast(blast_path)

        # Keep the best BLAST alignment per accession.
        best_by_acc = {}

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

            if acc not in best_by_acc or row["bitscore"] > best_by_acc[acc]["bitscore"]:
                best_by_acc[acc] = row

        fasta_records = []
        mutation_rows = []

        for acc, row in best_by_acc.items():
            seq = clean_subject_sequence(row["sseq"])

            if not seq:
                continue

            mapped_any = False

            for mut_name, qpos, wt, alt in mutations:
                mapped = map_query_position_to_extracted_subject_position(
                    row["qseq"],
                    row["sseq"],
                    row["qstart"],
                    qpos,
                )

                if mapped is None:
                    continue

                extracted_pos, query_residue, subject_residue = mapped

                if query_residue != wt:
                    continue

                # Only allow mutation if homolog has the expected WT residue
                # at the aligned coordinate.
                if subject_residue != wt:
                    continue

                # Sanity check against extracted sequence.
                if seq[extracted_pos - 1] != wt:
                    continue

                mapped_any = True

                mutation_rows.append({
                    "gene": gene,
                    "accession": acc,
                    "mutation": mut_name,
                    "query_pos": qpos,
                    # We intentionally call this subject_pos so the later
                    # injection script can use the same field name.
                    # It means position in the extracted ungapped BLAST subject sequence.
                    "subject_pos": extracted_pos,
                    "wt": wt,
                    "alt": alt,
                    "query_residue": query_residue,
                    "subject_residue": subject_residue,
                    "pident": row["pident"],
                    "qcovs": row["qcovs"],
                    "qlen": row["qlen"],
                    "slen": row["slen"],
                    "extracted_seq_len": len(seq),
                    "bitscore": row["bitscore"],
                    "stitle": row["stitle"],
                })

            if mapped_any:
                fasta_records.append((acc, seq))

        fasta_path = fasta_dir / f"{gene}_blast_derived_homologs.faa"
        write_fasta(fasta_records, fasta_path)

        map_path = map_dir / f"{gene}_mutation_map.tsv"
        with open(map_path, "w", newline="") as out:
            fieldnames = [
                "gene", "accession", "mutation", "query_pos", "subject_pos",
                "wt", "alt", "query_residue", "subject_residue",
                "pident", "qcovs", "qlen", "slen", "extracted_seq_len",
                "bitscore", "stitle"
            ]
            writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(mutation_rows)

        print(f"{gene}:")
        print(f"  best filtered accessions: {len(best_by_acc)}")
        print(f"  FASTA records written:    {len(fasta_records)}")
        print(f"  mutation-map rows:        {len(mutation_rows)}")
        print(f"  wrote FASTA:              {fasta_path}")
        print(f"  wrote mutation map:       {map_path}")

if __name__ == "__main__":
    main()
