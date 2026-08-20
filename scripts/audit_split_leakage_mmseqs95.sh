#!/usr/bin/env bash
set -euo pipefail

DATASET="$1"

BASEDIR="data/final_cluster95/${DATASET}"
OUTDIR="results/leakage_audit95/${DATASET}"
TMPDIR="tmp/leakage_audit95/${DATASET}"

mkdir -p "$OUTDIR" "$TMPDIR"

for split in train val test; do
    mmseqs createdb "${BASEDIR}/${split}.faa" "${OUTDIR}/${split}_DB"
done

run_audit () {
    qsplit="$1"
    tsplit="$2"

    echo "==== Auditing ${DATASET}: ${qsplit} vs ${tsplit} at 95% ===="

    mmseqs search \
        "${OUTDIR}/${qsplit}_DB" \
        "${OUTDIR}/${tsplit}_DB" \
        "${OUTDIR}/${qsplit}_vs_${tsplit}_search" \
        "${TMPDIR}/${qsplit}_vs_${tsplit}" \
        --min-seq-id 0.95 \
        -c 0.80 \
        --cov-mode 0

    mmseqs convertalis \
        "${OUTDIR}/${qsplit}_DB" \
        "${OUTDIR}/${tsplit}_DB" \
        "${OUTDIR}/${qsplit}_vs_${tsplit}_search" \
        "${OUTDIR}/${qsplit}_vs_${tsplit}.tsv" \
        --format-output "query,target,pident,alnlen,qcov,tcov,evalue,bits"

    echo "${qsplit}_vs_${tsplit}: $(wc -l < ${OUTDIR}/${qsplit}_vs_${tsplit}.tsv) hits"
}

run_audit train val
run_audit train test
run_audit val test
