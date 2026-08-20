#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="hugohrban/progen2-small"

MUTATIONS=(
  F33I
  D58N
  Q72H
  A75V
  V85A
  V154F
  Y180F
)

OUTDIR="results/ale_site_scores/mutation_holdout"
mkdir -p "$OUTDIR"

for MUT in "${MUTATIONS[@]}"; do
  echo "============================================================"
  echo "Evaluating held-out mutation: ${MUT}"
  echo "============================================================"

  RUN_DIR="runs/mutation_holdout/progen2_151m_dgoa_holdout_${MUT}_l40"
  MAP="../data/mutation_holdout/dgoA/holdout_${MUT}/test_heldout_mapping.tsv"

  if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: Missing run directory: $RUN_DIR"
    exit 1
  fi

  if [[ ! -f "$MAP" ]]; then
    echo "ERROR: Missing held-out mapping table: $MAP"
    exit 1
  fi

  # Prefer final adapter at run root if present; otherwise use latest checkpoint.
  if [[ -f "${RUN_DIR}/adapter_config.json" || -f "${RUN_DIR}/adapter_model.safetensors" || -f "${RUN_DIR}/adapter_model.bin" ]]; then
    FT_MODEL="$RUN_DIR"
  else
    FT_MODEL="$(find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  fi

  if [[ -z "${FT_MODEL:-}" || ! -d "$FT_MODEL" ]]; then
    echo "ERROR: Could not find fine-tuned model/checkpoint for ${MUT}"
    exit 1
  fi

  echo "Using fine-tuned model: $FT_MODEL"
  echo "Using mapping table:      $MAP"

  PRE_OUT="${OUTDIR}/pretrained_progen2_151m_dgoa_holdout_${MUT}_test_sites.csv"
  FT_OUT="${OUTDIR}/finetuned_progen2_151m_dgoa_holdout_${MUT}_test_sites.csv"
  CMP_PREFIX="${OUTDIR}/progen2_151m_dgoa_holdout_${MUT}_pretrained_vs_finetuned_test_sites"

  python ../scripts/eval/score_test_homolog_sites_causal.py \
    --model "$BASE_MODEL" \
    --model_label pretrained_progen2_151m \
    --mapping_tsv "$MAP" \
    --out "$PRE_OUT" \
    --dataset_filter dgoA \
    --batch_size 4

  python ../scripts/eval/score_test_homolog_sites_causal.py \
    --model "$FT_MODEL" \
    --base_model "$BASE_MODEL" \
    --model_label "finetuned_progen2_151m_dgoa_holdout_${MUT}_l40" \
    --mapping_tsv "$MAP" \
    --out "$FT_OUT" \
    --dataset_filter dgoA \
    --batch_size 4

  python ../scripts/eval/compare_pretrained_vs_finetuned_site_scores.py \
    --pretrained_csv "$PRE_OUT" \
    --finetuned_csv "$FT_OUT" \
    --out_prefix "$CMP_PREFIX" \
    --comparison_label "progen2_151m_dgoa_holdout_${MUT}_l40"

  echo "Done: ${MUT}"
done

echo "============================================================"
echo "All held-out mutation evaluations complete."
echo "Outputs are in: ${OUTDIR}"
echo "============================================================"
