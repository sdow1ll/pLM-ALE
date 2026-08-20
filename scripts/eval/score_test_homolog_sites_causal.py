#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    records = {}
    header = None
    seq_lines = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(seq_lines).upper()
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            records[header] = "".join(seq_lines).upper()

    return records


def load_all_test_fastas():
    records = {}

    paths = {
        "ecoli": Path("data/final_cluster95/ecoli/test.faa"),
        "dgoA": Path("data/final_cluster95/dgoA/test.faa"),
    }

    for dataset, path in paths.items():
        if not path.exists():
            continue

        fasta_records = read_fasta(path)

        for header, seq in fasta_records.items():
            records[(dataset, header)] = seq

    return records


def infer_base_model_from_adapter(adapter_dir):
    adapter_config = Path(adapter_dir) / "adapter_config.json"

    if not adapter_config.exists():
        return None

    with open(adapter_config) as f:
        cfg = json.load(f)

    return cfg.get("base_model_name_or_path")


def load_model(model_name_or_path, base_model=None, device="cuda"):
    model_path = Path(model_name_or_path)
    adapter_config = model_path / "adapter_config.json"

    is_adapter = adapter_config.exists()

    if is_adapter:
        if PeftModel is None:
            raise RuntimeError("This looks like a PEFT/LoRA adapter, but peft is not installed.")

        if base_model is None:
            base_model = infer_base_model_from_adapter(model_name_or_path)

        if base_model is None:
            raise RuntimeError(
                "Could not infer base model from adapter_config.json. "
                "Please provide --base_model."
            )

        print(f"Loading tokenizer from base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        print(f"Loading base causal LM: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)

        print(f"Loading LoRA adapter: {model_name_or_path}")
        model = PeftModel.from_pretrained(model, model_name_or_path)
    else:
        print(f"Loading tokenizer/model: {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "right"

    model.eval()
    model.to(device)

    return tokenizer, model


def aa_token_id(tokenizer, aa):
    ids = tokenizer.encode(aa, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Residue {aa} did not map to exactly one token: {ids}")
    return ids[0]


def rank_from_logits(logits, token_id):
    sorted_ids = torch.argsort(logits, descending=True)
    return (sorted_ids == token_id).nonzero(as_tuple=True)[0].item() + 1


@torch.no_grad()
def score_batch(tokenizer, model, batch_rows, seq_records, aa_token_ids, device):
    prefixes = []
    metadata = []

    for row in batch_rows:
        dataset = row["dataset"]
        header = row["fasta_header"]
        seq = seq_records[(dataset, header)]

        mapped_pos = int(row["homolog_mapped_pos_1based"])
        idx = mapped_pos - 1

        if idx < 0 or idx >= len(seq):
            raise ValueError(f"Mapped position {mapped_pos} outside sequence length {len(seq)}")

        wt = row["wt"]
        alt = row["alt"]
        mutation = row["mutation"]

        observed = seq[idx]

        # Your test FASTAs are mutated, so the residue at mapped_pos should be the ALE mutant.
        if observed != alt:
            raise ValueError(
                f"Unexpected residue for {dataset} {row['gene']} {row['accession']} {mutation}: "
                f"expected mutant {alt} at mapped position {mapped_pos}, observed {observed}"
            )

        # For a causal LM, probability of residue i is predicted from left context seq[:i].
        prefix = seq[:idx]

        if len(prefix) == 0:
            raise ValueError(
                f"Cannot score position 1 with this simple causal script because there is no left context: "
                f"{dataset} {row['gene']} {row['accession']} {mutation}"
            )

        prefixes.append(prefix)

        metadata.append({
            "seq": seq,
            "observed": observed,
            "mapped_pos": mapped_pos,
            "wt": wt,
            "alt": alt,
        })

    encoded = tokenizer(
        prefixes,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    outputs = model(**encoded)
    logits_all = outputs.logits

    results = []

    for i, row in enumerate(batch_rows):
        attention_mask = encoded["attention_mask"][i]

        # Last real prefix token predicts the next residue.
        last_prefix_token_pos = int(attention_mask.sum().item()) - 1

        logits = logits_all[i, last_prefix_token_pos]
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        wt = metadata[i]["wt"]
        alt = metadata[i]["alt"]

        wt_id = aa_token_ids[wt]
        alt_id = aa_token_ids[alt]

        logp_wt = log_probs[wt_id].item()
        logp_alt = log_probs[alt_id].item()
        p_wt = probs[wt_id].item()
        p_alt = probs[alt_id].item()

        aa_probs = {}
        aa_logps = {}
        aa_ranks = {}

        for aa in AA_ORDER:
            tid = aa_token_ids[aa]
            aa_probs[f"p_{aa}"] = probs[tid].item()
            aa_logps[f"logp_{aa}"] = log_probs[tid].item()
            aa_ranks[f"rank_{aa}"] = rank_from_logits(logits, tid)

        top_aa = max(AA_ORDER, key=lambda aa: aa_probs[f"p_{aa}"])

        result = {
            "model_label": row["model_label"],
            "dataset": row["dataset"],
            "split": row["split"],
            "gene": row["gene"],
            "accession": row["accession"],
            "mutation": row["mutation"],
            "wt": wt,
            "alt": alt,
            "reference_pos_1based": int(row["reference_pos_1based"]),
            "homolog_mapped_pos_1based": int(row["homolog_mapped_pos_1based"]),
            "observed_residue_in_test_sequence": metadata[i]["observed"],
            "sequence_length": int(row["sequence_length"]),
            "cluster": row["cluster"],
            "p_wt": p_wt,
            "p_alt": p_alt,
            "logp_wt": logp_wt,
            "logp_alt": logp_alt,
            "llr_alt_minus_wt": logp_alt - logp_wt,
            "wt_rank": rank_from_logits(logits, wt_id),
            "alt_rank": rank_from_logits(logits, alt_id),
            "top_aa": top_aa,
            "top_aa_prob": aa_probs[f"p_{top_aa}"],
            **aa_probs,
            **aa_logps,
            **aa_ranks,
        }

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Pretrained causal LM or LoRA adapter directory")
    parser.add_argument("--base_model", default=None, help="Optional base model for LoRA adapter")
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--mapping_tsv", default="results/ale_site_scores/test_homolog_mapped_sites.tsv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset_filter", choices=["ecoli", "dgoA"], default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device = "cpu"

    print(f"Using device: {device}")

    tokenizer, model = load_model(
        model_name_or_path=args.model,
        base_model=args.base_model,
        device=device,
    )

    aa_token_ids = {aa: aa_token_id(tokenizer, aa) for aa in AA_ORDER}

    mapping = pd.read_csv(args.mapping_tsv, sep="\t")

    if args.dataset_filter is not None:
        mapping = mapping[mapping["dataset"] == args.dataset_filter].copy()

    if len(mapping) == 0:
        raise RuntimeError("No rows left after filtering mapping TSV.")

    mapping["model_label"] = args.model_label

    seq_records = load_all_test_fastas()

    rows = mapping.to_dict(orient="records")
    all_results = []

    for start in tqdm(range(0, len(rows), args.batch_size), desc="Scoring"):
        batch_rows = rows[start:start + args.batch_size]

        batch_results = score_batch(
            tokenizer=tokenizer,
            model=model,
            batch_rows=batch_rows,
            seq_records=seq_records,
            aa_token_ids=aa_token_ids,
            device=device,
        )

        all_results.extend(batch_results)

    out_df = pd.DataFrame(all_results)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print()
    print(out_df[[
        "model_label",
        "dataset",
        "gene",
        "mutation",
        "accession",
        "p_wt",
        "p_alt",
        "logp_wt",
        "logp_alt",
        "llr_alt_minus_wt",
        "wt_rank",
        "alt_rank",
        "top_aa",
        "top_aa_prob",
    ]].head(20))

    print()
    print("Summary by dataset/gene/mutation:")
    print(
        out_df.groupby(["dataset", "gene", "mutation"])["llr_alt_minus_wt"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    print()
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
