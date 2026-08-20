#!/usr/bin/env bash
set -euo pipefail

cd /project/massrl/silbadowell

THREADS="${THREADS:-32}"
mkdir -p reports/verify

summary="reports/verify/leakage_audit95_summary.tsv"
echo -e "dataset\tpair\thits" > "$summary"

for dataset in ecoli dgoA; do
    BASEDIR="data/final_cluster95/${dataset}"
    OUTDIR="results/leakage_audit95/${dataset}"
    TMPDIR="tmp/leakage_audit95/${dataset}"

    mkdir -p "$OUTDIR" "$TMPDIR"

    echo "==== Building MMseqs DBs for ${dataset} ===="

    for split in train val test; do
        rm -f "${OUTDIR}/${split}_DB"*
        mmseqs createdb "${BASEDIR}/${split}.faa" "${OUTDIR}/${split}_DB"
    done

    for pair in train:val train:test val:test; do
        qsplit="${pair%%:*}"
        tsplit="${pair##*:}"
        pair_name="${qsplit}_vs_${tsplit}"

        echo "==== Auditing ${dataset}: ${pair_name} at 95% ===="

        rm -f "${OUTDIR}/${pair_name}_search"*
        rm -rf "${TMPDIR}/${pair_name}"

        mmseqs search \
            "${OUTDIR}/${qsplit}_DB" \
            "${OUTDIR}/${tsplit}_DB" \
            "${OUTDIR}/${pair_name}_search" \
            "${TMPDIR}/${pair_name}" \
            --min-seq-id 0.95 \
            -c 0.80 \
            --cov-mode 0 \
            --threads "$THREADS"

        mmseqs convertalis \
            "${OUTDIR}/${qsplit}_DB" \
            "${OUTDIR}/${tsplit}_DB" \
            "${OUTDIR}/${pair_name}_search" \
            "${OUTDIR}/${pair_name}.tsv" \
            --format-output "query,target,pident,alnlen,qcov,tcov,evalue,bits"

        hits=$(wc -l < "${OUTDIR}/${pair_name}.tsv")
        echo -e "${dataset}\t${pair_name}\t${hits}" >> "$summary"
        echo "${dataset} ${pair_name}: ${hits} hits"
    done
done

echo
cat "$summary"
echo
echo "Wrote: $summary"
