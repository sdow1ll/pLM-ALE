#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw/homologs tmp/accession_chunks logs

CHUNK_SIZE=10
MAX_RETRIES=5

for gene in topA yeiB spoT dgoA; do
    echo "==== Retrieving eligible homologs for ${gene} using direct efetch ===="

    acc_file="data/processed/mutation_maps/${gene}_eligible_accessions.txt"
    out_file="data/raw/homologs/${gene}_eligible_homologs.faa"
    chunk_dir="tmp/accession_chunks/${gene}"
    log_file="logs/${gene}_efetch_failures.log"

    rm -rf "$chunk_dir"
    mkdir -p "$chunk_dir"
    rm -f "$out_file" "$log_file"

    split -l "$CHUNK_SIZE" "$acc_file" "${chunk_dir}/chunk_"

    for chunk in "${chunk_dir}"/chunk_*; do
        ids=$(paste -sd, "$chunk")
        chunk_out="${chunk}.faa"

        echo "Retrieving ${chunk}"

        success=0

        for attempt in $(seq 1 "$MAX_RETRIES"); do
            echo "  attempt ${attempt}/${MAX_RETRIES}"

            if efetch -db protein -id "$ids" -format fasta > "$chunk_out"; then
                if grep -q "^>" "$chunk_out"; then
                    success=1
                    break
                fi
            fi

            sleep $((attempt * 5))
        done

        if [ "$success" -eq 1 ]; then
            cat "$chunk_out" >> "$out_file"
        else
            echo "FAILED ${chunk}" | tee -a "$log_file"
            cat "$chunk" >> "$log_file"
        fi

        sleep 1
    done

    echo "${gene}: $(grep -c '^>' "$out_file") sequences written to $out_file"

    if [ -f "$log_file" ]; then
        echo "WARNING: Some chunks failed for ${gene}. See $log_file"
    fi
done
