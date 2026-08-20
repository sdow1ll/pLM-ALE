#!/usr/bin/env bash
set -euo pipefail

cd /project/massrl/silbadowell

echo "==== E. coli token metrics and pseudo-perplexity analysis ===="
echo "Start: $(date)"
echo "Host: $(hostname)"
echo

FASTA="data/final_cluster95/ecoli/test.faa"

ESM2_BASE="facebook/esm2_t30_150M_UR50D"
ESM2_FT="pLM-ALE/runs/l40/esm2_150m_ecoli_cluster95_l40"

PROGEN2_BASE="hugohrban/progen2-small"
PROGEN2_FT="pLM-ALE/runs/l40/progen2_151m_ecoli_cluster95_l40"

OUTDIR="results/token_metrics/ecoli_cluster95"
mkdir -p "${OUTDIR}"

MLM_MASK_BATCH_SIZE="${MLM_MASK_BATCH_SIZE:-32}"
CAUSAL_BATCH_SIZE="${CAUSAL_BATCH_SIZE:-4}"
FORCE="${FORCE:-0}"

echo "FASTA: ${FASTA}"
echo "ESM-2 base: ${ESM2_BASE}"
echo "ESM-2 fine-tuned: ${ESM2_FT}"
echo "ProGen2 base: ${PROGEN2_BASE}"
echo "ProGen2 fine-tuned: ${PROGEN2_FT}"
echo "MLM mask batch size: ${MLM_MASK_BATCH_SIZE}"
echo "Causal batch size: ${CAUSAL_BATCH_SIZE}"
echo

if [[ ! -f "${FASTA}" ]]; then
    echo "ERROR: Missing FASTA: ${FASTA}"
    exit 1
fi

for d in "${ESM2_FT}" "${PROGEN2_FT}"; do
    if [[ ! -d "$d" ]]; then
        echo "ERROR: Missing fine-tuned model dir: $d"
        exit 1
    fi
    if [[ ! -f "$d/adapter_config.json" ]]; then
        echo "ERROR: Missing adapter_config.json in $d"
        exit 1
    fi
done

echo "Test sequences: $(grep -c '^>' "${FASTA}")"
echo

run_if_needed() {
    local json_file="$1"
    shift

    if [[ "${FORCE}" != "1" && -f "${json_file}" ]]; then
        echo "Skipping existing: ${json_file}"
    else
        "$@"
    fi
}

echo "==== 1. ESM-2 pretrained masked-token metrics and pseudo-perplexity ===="
run_if_needed \
    "${OUTDIR}/esm2_pretrained.json" \
    python scripts/eval/compute_token_metrics_and_ppl.py \
        --model_type mlm \
        --model "${ESM2_BASE}" \
        --model_label esm2_pretrained \
        --fasta "${FASTA}" \
        --out_prefix "${OUTDIR}/esm2_pretrained" \
        --mask_batch_size "${MLM_MASK_BATCH_SIZE}" \
        --batch_size 1 \
        --sequence_format raw

echo

echo "==== 2. ESM-2 fine-tuned masked-token metrics and pseudo-perplexity ===="
run_if_needed \
    "${OUTDIR}/esm2_finetuned.json" \
    python scripts/eval/compute_token_metrics_and_ppl.py \
        --model_type mlm \
        --model "${ESM2_FT}" \
        --base_model "${ESM2_BASE}" \
        --model_label esm2_finetuned \
        --fasta "${FASTA}" \
        --out_prefix "${OUTDIR}/esm2_finetuned" \
        --mask_batch_size "${MLM_MASK_BATCH_SIZE}" \
        --batch_size 1 \
        --sequence_format raw

echo

echo "==== 3. ProGen2 pretrained next-token metrics and perplexity ===="
run_if_needed \
    "${OUTDIR}/progen2_pretrained.json" \
    python scripts/eval/compute_token_metrics_and_ppl.py \
        --model_type causal \
        --model "${PROGEN2_BASE}" \
        --model_label progen2_pretrained \
        --fasta "${FASTA}" \
        --out_prefix "${OUTDIR}/progen2_pretrained" \
        --batch_size "${CAUSAL_BATCH_SIZE}" \
        --sequence_format auto

echo

echo "==== 4. ProGen2 fine-tuned next-token metrics and perplexity ===="
run_if_needed \
    "${OUTDIR}/progen2_finetuned.json" \
    python scripts/eval/compute_token_metrics_and_ppl.py \
        --model_type causal \
        --model "${PROGEN2_FT}" \
        --base_model "${PROGEN2_BASE}" \
        --model_label progen2_finetuned \
        --fasta "${FASTA}" \
        --out_prefix "${OUTDIR}/progen2_finetuned" \
        --batch_size "${CAUSAL_BATCH_SIZE}" \
        --sequence_format auto

echo

echo "==== 5. Generate CSV and LaTeX tables ===="
python scripts/eval/make_token_metric_tables.py \
    --esm2_pretrained_json "${OUTDIR}/esm2_pretrained.json" \
    --esm2_finetuned_json "${OUTDIR}/esm2_finetuned.json" \
    --progen2_pretrained_json "${OUTDIR}/progen2_pretrained.json" \
    --progen2_finetuned_json "${OUTDIR}/progen2_finetuned.json" \
    --out_prefix "${OUTDIR}/ecoli_cluster95"

echo

echo "==== Output files ===="
ls -lh "${OUTDIR}"

echo
echo "==== Done ===="
echo "End: $(date)"
