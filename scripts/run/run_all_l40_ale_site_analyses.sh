#!/usr/bin/env bash
set -euo pipefail

cd /project/massrl/silbadowell

echo "==== Running ALE-site analyses for all L40 models ===="
echo "Start: $(date)"
echo "Host: $(hostname)"
echo

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

# ProGen2 DgoA saved only checkpoints, so use latest checkpoint.
PROGEN2_DGOA_RUN="pLM-ALE/runs/l40/progen2_151m_dgoa_cluster95_l40"
PROGEN2_DGOA_FT="$(find "${PROGEN2_DGOA_RUN}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"

# ----------------------------
# Inputs
# ----------------------------
MAPPING_TSV="results/ale_site_scores/test_homolog_mapped_sites.tsv"
ECOLI_MAPPING_TSV="results/ale_site_scores/test_homolog_mapped_sites_ecoli.tsv"
DGOA_MAPPING_TSV="results/ale_site_scores/test_homolog_mapped_sites_dgoA.tsv"

ESM_BATCH_SIZE="${ESM_BATCH_SIZE:-8}"
PROGEN_BATCH_SIZE="${PROGEN_BATCH_SIZE:-4}"

mkdir -p results/ale_site_scores
mkdir -p results/figures/ale_site_scores
mkdir -p results/figures/window_llr_heatmaps

echo "==== Model paths ===="
echo "ESM2 base:          ${ESM2_BASE}"
echo "ProGen2 base:       ${PROGEN2_BASE}"
echo "ESM2 E. coli FT:    ${ESM2_ECOLI_FT}"
echo "ESM2 DgoA FT:       ${ESM2_DGOA_FT}"
echo "ProGen2 E. coli FT: ${PROGEN2_ECOLI_FT}"
echo "ProGen2 DgoA FT:    ${PROGEN2_DGOA_FT}"
echo

# ----------------------------
# Checks
# ----------------------------
echo "==== Checking required files and directories ===="

for f in \
  "${MAPPING_TSV}" \
  scripts/eval/score_test_homolog_sites_mlm.py \
  scripts/eval/score_test_homolog_sites_causal.py \
  scripts/eval/compare_pretrained_vs_finetuned_site_scores.py \
  scripts/plotting/plot_ale_site_llr_figures.py \
  scripts/plotting/plot_llr_heatmap_from_site_scores.py \
  scripts/plotting/plot_window_llr_heatmap_model_pair.py
do
  if [[ ! -e "$f" ]]; then
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

echo "Checks passed."
echo

# ----------------------------
# Create filtered mapping TSVs for ESM-2 MLM scorer
# because that script does not support --dataset_filter.
# ----------------------------
echo "==== Creating dataset-specific mapping TSVs ===="

python - <<PY
import pandas as pd

src = "${MAPPING_TSV}"
df = pd.read_csv(src, sep="\t")

for dataset, out in [
    ("ecoli", "${ECOLI_MAPPING_TSV}"),
    ("dgoA", "${DGOA_MAPPING_TSV}"),
]:
    sub = df[df["dataset"] == dataset].copy()
    sub.to_csv(out, sep="\t", index=False)
    print(f"Wrote {out}: {len(sub)} rows")
    print(sub.groupby(["gene", "mutation"]).size().to_string())
    print()
PY

# ----------------------------
# Helper functions
# ----------------------------
score_esm2() {
  local dataset="$1"
  local mapping="$2"
  local ft_model="$3"
  local pretrained_out="$4"
  local finetuned_out="$5"
  local compare_prefix="$6"
  local fig_prefix="$7"
  local title_prefix="$8"

  echo
  echo "============================================================"
  echo "ESM-2 ${dataset}"
  echo "============================================================"

  echo "==== Score pretrained ESM-2 ===="
  python scripts/eval/score_test_homolog_sites_mlm.py \
    --model "${ESM2_BASE}" \
    --model_label pretrained_esm2_150m \
    --mapping_tsv "${mapping}" \
    --out "${pretrained_out}" \
    --batch_size "${ESM_BATCH_SIZE}"

  echo "==== Score fine-tuned ESM-2 ===="
  python scripts/eval/score_test_homolog_sites_mlm.py \
    --model "${ft_model}" \
    --base_model "${ESM2_BASE}" \
    --model_label "finetuned_esm2_150m_${dataset}_cluster95_l40" \
    --mapping_tsv "${mapping}" \
    --out "${finetuned_out}" \
    --batch_size "${ESM_BATCH_SIZE}"

  echo "==== Compare pretrained vs fine-tuned ===="
  python scripts/eval/compare_pretrained_vs_finetuned_site_scores.py \
    --pretrained_csv "${pretrained_out}" \
    --finetuned_csv "${finetuned_out}" \
    --out_prefix "${compare_prefix}" \
    --comparison_label "esm2_150m_${dataset}_cluster95_l40"

  echo "==== Summary ===="
  python - <<PY
import pandas as pd
f = "${compare_prefix}_summary.csv"
df = pd.read_csv(f)
cols = [
    "mutation", "n",
    "mean_pretrained_llr", "mean_finetuned_llr", "mean_delta_llr",
    "median_delta_llr",
    "mean_pretrained_p_alt", "mean_finetuned_p_alt", "mean_delta_p_alt",
    "frac_delta_llr_positive", "frac_alt_rank_improved",
]
cols = [c for c in cols if c in df.columns]
print(df[cols].to_string(index=False))
PY

  echo "==== Main/supporting/supplementary LLR figures ===="
  python scripts/plotting/plot_ale_site_llr_figures.py \
    --summary_csv "${compare_prefix}_summary.csv" \
    --full_csv "${compare_prefix}_full.csv" \
    --out_prefix "${fig_prefix}" \
    --title_prefix "${title_prefix}"

  echo "==== Amino-acid LLR heatmaps with numeric true-mutant values ===="
  python scripts/plotting/plot_llr_heatmap_from_site_scores.py \
    --comparison_full_csv "${compare_prefix}_full.csv" \
    --out_prefix "${fig_prefix}" \
    --dataset_filter "${dataset}"
}

score_progen2() {
  local dataset="$1"
  local ft_model="$2"
  local pretrained_out="$3"
  local finetuned_out="$4"
  local compare_prefix="$5"
  local fig_prefix="$6"
  local title_prefix="$7"

  echo
  echo "============================================================"
  echo "ProGen2 ${dataset}"
  echo "============================================================"

  echo "==== Score pretrained ProGen2 ===="
  python scripts/eval/score_test_homolog_sites_causal.py \
    --model "${PROGEN2_BASE}" \
    --model_label pretrained_progen2_151m \
    --mapping_tsv "${MAPPING_TSV}" \
    --out "${pretrained_out}" \
    --dataset_filter "${dataset}" \
    --batch_size "${PROGEN_BATCH_SIZE}"

  echo "==== Score fine-tuned ProGen2 ===="
  python scripts/eval/score_test_homolog_sites_causal.py \
    --model "${ft_model}" \
    --base_model "${PROGEN2_BASE}" \
    --model_label "finetuned_progen2_151m_${dataset}_cluster95_l40" \
    --mapping_tsv "${MAPPING_TSV}" \
    --out "${finetuned_out}" \
    --dataset_filter "${dataset}" \
    --batch_size "${PROGEN_BATCH_SIZE}"

  echo "==== Compare pretrained vs fine-tuned ===="
  python scripts/eval/compare_pretrained_vs_finetuned_site_scores.py \
    --pretrained_csv "${pretrained_out}" \
    --finetuned_csv "${finetuned_out}" \
    --out_prefix "${compare_prefix}" \
    --comparison_label "progen2_151m_${dataset}_cluster95_l40"

  echo "==== Summary ===="
  python - <<PY
import pandas as pd
f = "${compare_prefix}_summary.csv"
df = pd.read_csv(f)
cols = [
    "mutation", "n",
    "mean_pretrained_llr", "mean_finetuned_llr", "mean_delta_llr",
    "median_delta_llr",
    "mean_pretrained_p_alt", "mean_finetuned_p_alt", "mean_delta_p_alt",
    "frac_delta_llr_positive", "frac_alt_rank_improved",
]
cols = [c for c in cols if c in df.columns]
print(df[cols].to_string(index=False))
PY

  echo "==== Main/supporting/supplementary LLR figures ===="
  python scripts/plotting/plot_ale_site_llr_figures.py \
    --summary_csv "${compare_prefix}_summary.csv" \
    --full_csv "${compare_prefix}_full.csv" \
    --out_prefix "${fig_prefix}" \
    --title_prefix "${title_prefix}"

  echo "==== Amino-acid LLR heatmaps with numeric true-mutant values ===="
  python scripts/plotting/plot_llr_heatmap_from_site_scores.py \
    --comparison_full_csv "${compare_prefix}_full.csv" \
    --out_prefix "${fig_prefix}" \
    --dataset_filter "${dataset}"
}

run_window_heatmaps() {
  local model_type="$1"
  local base_model="$2"
  local ft_model="$3"
  local pretrained_label="$4"
  local finetuned_label="$5"
  local dataset="$6"
  local fig_prefix_root="$7"

  echo
  echo "==== Local window LLR heatmaps: ${fig_prefix_root} ${dataset} ===="

  if [[ "${dataset}" == "ecoli" ]]; then
    items=("topA H33Y" "yeiB L143I" "spoT K662I")
  else
    items=("dgoA F33I" "dgoA D58N" "dgoA Q72H" "dgoA A75V" "dgoA V85A" "dgoA V154F" "dgoA Y180F")
  fi

  for item in "${items[@]}"; do
    gene=$(echo "$item" | awk '{print $1}')
    mut=$(echo "$item" | awk '{print $2}')

    echo "---- ${gene} ${mut} ----"

    if [[ "${model_type}" == "mlm" ]]; then
      python scripts/plotting/plot_window_llr_heatmap_model_pair.py \
        --model_type mlm \
        --pretrained_model "${base_model}" \
        --finetuned_model "${ft_model}" \
        --finetuned_base_model "${base_model}" \
        --pretrained_label "${pretrained_label}" \
        --finetuned_label "${finetuned_label}" \
        --gene "${gene}" \
        --mutation "${mut}" \
        --window 10 \
        --out_prefix "results/figures/window_llr_heatmaps/${fig_prefix_root}_${dataset}_${gene}_${mut}_window10_clean"
    else
      python scripts/plotting/plot_window_llr_heatmap_model_pair.py \
        --model_type causal \
        --pretrained_model "${base_model}" \
        --finetuned_model "${ft_model}" \
        --finetuned_base_model "${base_model}" \
        --pretrained_label "${pretrained_label}" \
        --finetuned_label "${finetuned_label}" \
        --gene "${gene}" \
        --mutation "${mut}" \
        --window 10 \
        --out_prefix "results/figures/window_llr_heatmaps/${fig_prefix_root}_${dataset}_${gene}_${mut}_window10_clean"
    fi
  done
}

# ----------------------------
# ESM-2 E. coli
# ----------------------------
score_esm2 \
  ecoli \
  "${ECOLI_MAPPING_TSV}" \
  "${ESM2_ECOLI_FT}" \
  results/ale_site_scores/pretrained_esm2_150m_ecoli_test_homolog_sites.csv \
  results/ale_site_scores/finetuned_esm2_150m_ecoli_cluster95_l40_test_homolog_sites.csv \
  results/ale_site_scores/esm2_150m_ecoli_pretrained_vs_finetuned_test_homolog_sites \
  results/figures/ale_site_scores/esm2_150m_ecoli \
  "ESM-2 E. coli"

# ----------------------------
# ESM-2 DgoA
# ----------------------------
score_esm2 \
  dgoA \
  "${DGOA_MAPPING_TSV}" \
  "${ESM2_DGOA_FT}" \
  results/ale_site_scores/pretrained_esm2_150m_dgoa_test_homolog_sites.csv \
  results/ale_site_scores/finetuned_esm2_150m_dgoa_cluster95_l40_test_homolog_sites.csv \
  results/ale_site_scores/esm2_150m_dgoa_pretrained_vs_finetuned_test_homolog_sites \
  results/figures/ale_site_scores/esm2_150m_dgoa \
  "ESM-2 DgoA"

# ----------------------------
# ProGen2 E. coli
# ----------------------------
score_progen2 \
  ecoli \
  "${PROGEN2_ECOLI_FT}" \
  results/ale_site_scores/pretrained_progen2_151m_ecoli_test_homolog_sites.csv \
  results/ale_site_scores/finetuned_progen2_151m_ecoli_cluster95_l40_test_homolog_sites.csv \
  results/ale_site_scores/progen2_151m_ecoli_pretrained_vs_finetuned_test_homolog_sites \
  results/figures/ale_site_scores/progen2_151m_ecoli \
  "ProGen2 E. coli"

# ----------------------------
# ProGen2 DgoA
# ----------------------------
score_progen2 \
  dgoA \
  "${PROGEN2_DGOA_FT}" \
  results/ale_site_scores/pretrained_progen2_151m_dgoa_test_homolog_sites.csv \
  results/ale_site_scores/finetuned_progen2_151m_dgoa_cluster95_l40_test_homolog_sites.csv \
  results/ale_site_scores/progen2_151m_dgoa_pretrained_vs_finetuned_test_homolog_sites \
  results/figures/ale_site_scores/progen2_151m_dgoa \
  "ProGen2 DgoA"

# ----------------------------
# Local window heatmaps
# ----------------------------
run_window_heatmaps \
  mlm \
  "${ESM2_BASE}" \
  "${ESM2_ECOLI_FT}" \
  "ESM-2 pretrained" \
  "ESM-2 fine-tuned" \
  ecoli \
  esm2

run_window_heatmaps \
  mlm \
  "${ESM2_BASE}" \
  "${ESM2_DGOA_FT}" \
  "ESM-2 pretrained" \
  "ESM-2 fine-tuned" \
  dgoA \
  esm2

run_window_heatmaps \
  causal \
  "${PROGEN2_BASE}" \
  "${PROGEN2_ECOLI_FT}" \
  "ProGen2 pretrained" \
  "ProGen2 fine-tuned" \
  ecoli \
  progen2

run_window_heatmaps \
  causal \
  "${PROGEN2_BASE}" \
  "${PROGEN2_DGOA_FT}" \
  "ProGen2 pretrained" \
  "ProGen2 fine-tuned" \
  dgoA \
  progen2

echo
echo "==== Final output check ===="
echo

echo "ALE-site CSV summaries:"
ls -lh results/ale_site_scores/*pretrained_vs_finetuned_test_homolog_sites_summary.csv

echo
echo "Delta heatmaps:"
ls -lh results/figures/ale_site_scores/*delta_llr_heatmap.png
ls -lh results/figures/ale_site_scores/*delta_llr_heatmap.pdf

echo
echo "Window heatmaps:"
ls -lh results/figures/window_llr_heatmaps/*window10_clean*.png 2>/dev/null || true
ls -lh results/figures/window_llr_heatmaps/*window10_clean*.pdf 2>/dev/null || true

echo
echo "==== Done ===="
echo "End: $(date)"
