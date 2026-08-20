#!/usr/bin/env bash
set -euo pipefail

cd /project/massrl/silbadowell

echo "==== L40 token metrics + pseudo-perplexity analysis ===="
echo "Start: $(date)"
echo "Host: $(hostname)"
echo

# ----------------------------
# Test FASTAs from new split
# ----------------------------
ECOLI_FASTA="data/final_cluster95/ecoli/test.faa"
DGOA_FASTA="data/final_cluster95/dgoA/test.faa"

# ----------------------------
# Base models
# ----------------------------
ESM2_BASE="facebook/esm2_t30_150M_UR50D"
PROGEN2_BASE="hugohrban/progen2-small"

# ----------------------------
# Fine-tuned model paths
# ----------------------------
ESM2_ECOLI_FT="pLM-ALE/runs/l40/esm2_150m_ecoli_cluster95_l40"
ESM2_DGOA_FT="pLM-ALE/runs/l40/esm2_150m_dgoa_cluster95_l40"

PROGEN2_ECOLI_FT="pLM-ALE/runs/l40/progen2_151m_ecoli_cluster95_l40"

PROGEN2_DGOA_RUN="pLM-ALE/runs/l40/progen2_151m_dgoa_cluster95_l40"
PROGEN2_DGOA_FT="$(find "${PROGEN2_DGOA_RUN}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"

# ----------------------------
# Runtime settings
# ----------------------------
MLM_MASK_BATCH_SIZE="${MLM_MASK_BATCH_SIZE:-32}"
CAUSAL_BATCH_SIZE="${CAUSAL_BATCH_SIZE:-4}"
FORCE="${FORCE:-0}"

echo "E. coli FASTA: ${ECOLI_FASTA}"
echo "DgoA FASTA:    ${DGOA_FASTA}"
echo
echo "ESM-2 base:          ${ESM2_BASE}"
echo "ProGen2 base:        ${PROGEN2_BASE}"
echo "ESM-2 E. coli FT:    ${ESM2_ECOLI_FT}"
echo "ESM-2 DgoA FT:       ${ESM2_DGOA_FT}"
echo "ProGen2 E. coli FT:  ${PROGEN2_ECOLI_FT}"
echo "ProGen2 DgoA FT:     ${PROGEN2_DGOA_FT}"
echo
echo "MLM mask batch size: ${MLM_MASK_BATCH_SIZE}"
echo "Causal batch size:   ${CAUSAL_BATCH_SIZE}"
echo "FORCE:               ${FORCE}"
echo

# ----------------------------
# Checks
# ----------------------------
for f in \
  "${ECOLI_FASTA}" \
  "${DGOA_FASTA}" \
  scripts/eval/compute_token_metrics_and_ppl.py \
  scripts/eval/make_token_metric_tables.py
do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Missing required file/script: $f"
    exit 1
  fi
done

for d in \
  "${ESM2_ECOLI_FT}" \
  "${ESM2_DGOA_FT}" \
  "${PROGEN2_ECOLI_FT}" \
  "${PROGEN2_DGOA_FT}"
do
  if [[ ! -d "$d" ]]; then
    echo "ERROR: Missing model directory: $d"
    exit 1
  fi
  if [[ ! -f "$d/adapter_config.json" ]]; then
    echo "ERROR: Missing adapter_config.json in: $d"
    exit 1
  fi
done

echo "E. coli test records: $(grep -c '^>' "${ECOLI_FASTA}")"
echo "DgoA test records:    $(grep -c '^>' "${DGOA_FASTA}")"
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

run_dataset() {
  local dataset="$1"
  local fasta="$2"
  local esm2_ft="$3"
  local progen2_ft="$4"

  local outdir="results/token_metrics/${dataset}_cluster95"
  mkdir -p "${outdir}"

  echo
  echo "============================================================"
  echo "Dataset: ${dataset}"
  echo "FASTA: ${fasta}"
  echo "Output: ${outdir}"
  echo "============================================================"
  echo

  echo "==== 1. ESM-2 pretrained masked-token metrics ===="
  run_if_needed \
    "${outdir}/esm2_pretrained.json" \
    python scripts/eval/compute_token_metrics_and_ppl.py \
      --model_type mlm \
      --model "${ESM2_BASE}" \
      --model_label esm2_pretrained \
      --fasta "${fasta}" \
      --out_prefix "${outdir}/esm2_pretrained" \
      --mask_batch_size "${MLM_MASK_BATCH_SIZE}" \
      --batch_size 1 \
      --sequence_format raw

  echo
  echo "==== 2. ESM-2 fine-tuned masked-token metrics ===="
  run_if_needed \
    "${outdir}/esm2_finetuned.json" \
    python scripts/eval/compute_token_metrics_and_ppl.py \
      --model_type mlm \
      --model "${esm2_ft}" \
      --base_model "${ESM2_BASE}" \
      --model_label esm2_finetuned \
      --fasta "${fasta}" \
      --out_prefix "${outdir}/esm2_finetuned" \
      --mask_batch_size "${MLM_MASK_BATCH_SIZE}" \
      --batch_size 1 \
      --sequence_format raw

  echo
  echo "==== 3. ProGen2 pretrained next-token metrics ===="
  run_if_needed \
    "${outdir}/progen2_pretrained.json" \
    python scripts/eval/compute_token_metrics_and_ppl.py \
      --model_type causal \
      --model "${PROGEN2_BASE}" \
      --model_label progen2_pretrained \
      --fasta "${fasta}" \
      --out_prefix "${outdir}/progen2_pretrained" \
      --batch_size "${CAUSAL_BATCH_SIZE}" \
      --sequence_format auto

  echo
  echo "==== 4. ProGen2 fine-tuned next-token metrics ===="
  run_if_needed \
    "${outdir}/progen2_finetuned.json" \
    python scripts/eval/compute_token_metrics_and_ppl.py \
      --model_type causal \
      --model "${progen2_ft}" \
      --base_model "${PROGEN2_BASE}" \
      --model_label progen2_finetuned \
      --fasta "${fasta}" \
      --out_prefix "${outdir}/progen2_finetuned" \
      --batch_size "${CAUSAL_BATCH_SIZE}" \
      --sequence_format auto

  echo
  echo "==== 5. Generate CSV + LaTeX tables for ${dataset} ===="
  python scripts/eval/make_token_metric_tables.py \
    --esm2_pretrained_json "${outdir}/esm2_pretrained.json" \
    --esm2_finetuned_json "${outdir}/esm2_finetuned.json" \
    --progen2_pretrained_json "${outdir}/progen2_pretrained.json" \
    --progen2_finetuned_json "${outdir}/progen2_finetuned.json" \
    --out_prefix "${outdir}/${dataset}_cluster95"

  echo
  echo "==== ${dataset} output files ===="
  ls -lh "${outdir}"
}

run_dataset \
  ecoli \
  "${ECOLI_FASTA}" \
  "${ESM2_ECOLI_FT}" \
  "${PROGEN2_ECOLI_FT}"

run_dataset \
  dgoA \
  "${DGOA_FASTA}" \
  "${ESM2_DGOA_FT}" \
  "${PROGEN2_DGOA_FT}"

echo
echo "==== Final outputs ===="
echo

echo "E. coli:"
ls -lh results/token_metrics/ecoli_cluster95

echo
echo "DgoA:"
ls -lh results/token_metrics/dgoA_cluster95

echo
echo "==== Done ===="
echo "End: $(date)"
