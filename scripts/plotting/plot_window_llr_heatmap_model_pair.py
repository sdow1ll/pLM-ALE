#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")


def read_single_fasta(path):
    header = None
    seq_lines = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    raise RuntimeError(f"Expected one FASTA record in {path}")
                header = line[1:]
            else:
                seq_lines.append(line)

    if header is None:
        raise RuntimeError(f"No FASTA record found in {path}")

    return header, "".join(seq_lines).upper()


def parse_mutation(mutation):
    m = re.match(r"^([A-Z])([0-9]+)([A-Z])$", mutation)
    if m is None:
        raise ValueError(f"Could not parse mutation: {mutation}")

    wt = m.group(1)
    pos = int(m.group(2))
    alt = m.group(3)

    return wt, pos, alt


def infer_base_model_from_adapter(adapter_dir):
    adapter_config = Path(adapter_dir) / "adapter_config.json"

    if not adapter_config.exists():
        return None

    with open(adapter_config) as f:
        cfg = json.load(f)

    return cfg.get("base_model_name_or_path")


def load_model(model_path, model_type, base_model=None, device="cuda"):
    model_path_obj = Path(model_path)
    is_adapter = (model_path_obj / "adapter_config.json").exists()

    if is_adapter:
        if PeftModel is None:
            raise RuntimeError("This looks like a PEFT/LoRA adapter, but peft is not installed.")

        if base_model is None:
            base_model = infer_base_model_from_adapter(model_path)

        if base_model is None:
            raise RuntimeError(
                f"Could not infer base model for adapter {model_path}. "
                "Provide --pretrained_base_model or --finetuned_base_model."
            )

        print(f"Loading tokenizer from base model: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        if model_type == "mlm":
            print(f"Loading base masked LM: {base_model}")
            model = AutoModelForMaskedLM.from_pretrained(base_model, trust_remote_code=True)
        elif model_type == "causal":
            print(f"Loading base causal LM: {base_model}")
            model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
        else:
            raise ValueError(model_type)

        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)

    else:
        print(f"Loading tokenizer/model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if model_type == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        elif model_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        else:
            raise ValueError(model_type)

    if model_type == "causal":
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


def aa_token_ids(tokenizer):
    ids = {}

    for aa in AA_ORDER:
        encoded = tokenizer.encode(aa, add_special_tokens=False)

        if len(encoded) != 1:
            raise ValueError(f"Amino acid {aa} did not map to one token: {encoded}")

        ids[aa] = encoded[0]

    return ids


@torch.no_grad()
def score_position_mlm(tokenizer, model, seq, pos_1based, aa_ids, device):
    idx = pos_1based - 1

    if idx < 0 or idx >= len(seq):
        raise ValueError(f"Position {pos_1based} outside sequence length {len(seq)}")

    masked_seq = seq[:idx] + tokenizer.mask_token + seq[idx + 1:]

    encoded = tokenizer(masked_seq, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    input_ids = encoded["input_ids"][0]
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]

    if len(mask_positions) != 1:
        raise RuntimeError(f"Expected one mask token, found {len(mask_positions)}")

    mask_pos = mask_positions.item()

    logits = model(**encoded).logits[0, mask_pos]
    log_probs = torch.log_softmax(logits, dim=-1)

    wt = seq[idx]
    wt_logp = log_probs[aa_ids[wt]].item()

    return {aa: log_probs[aa_ids[aa]].item() - wt_logp for aa in AA_ORDER}


@torch.no_grad()
def score_position_causal(tokenizer, model, seq, pos_1based, aa_ids, device):
    idx = pos_1based - 1

    if idx < 0 or idx >= len(seq):
        raise ValueError(f"Position {pos_1based} outside sequence length {len(seq)}")

    if idx == 0:
        raise ValueError(
            "This causal-LM script cannot score position 1 because there is no left context."
        )

    prefix = seq[:idx]

    encoded = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=False,
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    logits = model(**encoded).logits[0, -1]
    log_probs = torch.log_softmax(logits, dim=-1)

    wt = seq[idx]
    wt_logp = log_probs[aa_ids[wt]].item()

    return {aa: log_probs[aa_ids[aa]].item() - wt_logp for aa in AA_ORDER}


def score_window(tokenizer, model, model_type, seq, start, end, aa_ids, device):
    mat = []

    for aa in AA_ORDER:
        row = []

        for pos in range(start, end + 1):
            if model_type == "mlm":
                scores = score_position_mlm(tokenizer, model, seq, pos, aa_ids, device)
            else:
                scores = score_position_causal(tokenizer, model, seq, pos, aa_ids, device)

            row.append(scores[aa])

        mat.append(row)

    return np.array(mat, dtype=float)


def choose_text_color(value, vmin, vmax):
    midpoint = (vmax + vmin) / 2.0
    return "white" if value < midpoint else "black"


def plot_pair(
    pretrained_mat,
    finetuned_mat,
    positions,
    gene,
    mutation,
    wt,
    mut_pos,
    alt,
    pretrained_label,
    finetuned_label,
    out_prefix,
):
    values = np.concatenate([pretrained_mat.flatten(), finetuned_mat.flatten()])
    vmax = np.nanmax(np.abs(values))
    vmin = -vmax

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    panels = [
        (axes[0], pretrained_mat, pretrained_label),
        (axes[1], finetuned_mat, finetuned_label),
    ]

    alt_row = AA_ORDER.index(alt)
    mut_col = positions.index(mut_pos)

    im = None

    for ax, mat, title in panels:
        im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")

        ax.set_title(title)
        ax.set_xlabel("Position")
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions, rotation=90)

        ax.set_yticks(range(len(AA_ORDER)))
        ax.set_yticklabels(AA_ORDER)

        # Annotate only the numerical LLR value for the real ALE mutant residue
        # at the true mutation position. No star marker and no vertical line.
        value = mat[alt_row, mut_col]

        ax.text(
            mut_col,
            alt_row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=choose_text_color(value, vmin, vmax),
        )

    axes[0].set_ylabel("Residue")

    fig.suptitle(f"{gene} mutation {mutation} local LLR heatmap", fontsize=16)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03)
    cbar.set_label("LLR: log P(AA) - log P(WT)")

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    png = str(out_prefix) + ".png"
    pdf = str(out_prefix) + ".pdf"

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()

    print(f"Wrote: {png}")
    print(f"Wrote: {pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["mlm", "causal"])
    parser.add_argument("--pretrained_model", required=True)
    parser.add_argument("--finetuned_model", required=True)
    parser.add_argument("--pretrained_base_model", default=None)
    parser.add_argument("--finetuned_base_model", default=None)
    parser.add_argument("--pretrained_label", default="Pretrained")
    parser.add_argument("--finetuned_label", default="Fine-tuned")
    parser.add_argument("--gene", required=True)
    parser.add_argument("--mutation", required=True)
    parser.add_argument("--fasta_dir", default="data/raw/queries/by_gene")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device = "cpu"

    wt, mut_pos, alt = parse_mutation(args.mutation)

    fasta_path = Path(args.fasta_dir) / f"{args.gene}_query.faa"
    header, seq = read_single_fasta(fasta_path)

    observed_wt = seq[mut_pos - 1]

    if observed_wt != wt:
        raise ValueError(
            f"Reference WT mismatch for {args.gene} {args.mutation}: "
            f"expected {wt} at position {mut_pos}, observed {observed_wt}"
        )

    start = max(1, mut_pos - args.window)
    end = min(len(seq), mut_pos + args.window)
    positions = list(range(start, end + 1))

    print(f"Gene: {args.gene}")
    print(f"Mutation: {args.mutation}")
    print(f"Sequence: {header}")
    print(f"Sequence length: {len(seq)}")
    print(f"Window: {start}-{end}")
    print(f"Mutation center: position {mut_pos}, {wt}->{alt}")
    print(f"Device: {device}")

    print("\nLoading pretrained model...")
    pre_tok, pre_model = load_model(
        args.pretrained_model,
        args.model_type,
        base_model=args.pretrained_base_model,
        device=device,
    )

    pre_aa_ids = aa_token_ids(pre_tok)

    print("Scoring pretrained model...")
    pretrained_mat = score_window(
        pre_tok,
        pre_model,
        args.model_type,
        seq,
        start,
        end,
        pre_aa_ids,
        device,
    )

    del pre_model

    if device == "cuda":
        torch.cuda.empty_cache()

    print("\nLoading fine-tuned model...")
    fin_tok, fin_model = load_model(
        args.finetuned_model,
        args.model_type,
        base_model=args.finetuned_base_model,
        device=device,
    )

    fin_aa_ids = aa_token_ids(fin_tok)

    print("Scoring fine-tuned model...")
    finetuned_mat = score_window(
        fin_tok,
        fin_model,
        args.model_type,
        seq,
        start,
        end,
        fin_aa_ids,
        device,
    )

    plot_pair(
        pretrained_mat=pretrained_mat,
        finetuned_mat=finetuned_mat,
        positions=positions,
        gene=args.gene,
        mutation=args.mutation,
        wt=wt,
        mut_pos=mut_pos,
        alt=alt,
        pretrained_label=args.pretrained_label,
        finetuned_label=args.finetuned_label,
        out_prefix=args.out_prefix,
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pre_df = pd.DataFrame(pretrained_mat, index=AA_ORDER, columns=positions)
    fin_df = pd.DataFrame(finetuned_mat, index=AA_ORDER, columns=positions)
    delta_df = fin_df - pre_df

    pre_df.to_csv(str(out_prefix) + "_pretrained_matrix.csv")
    fin_df.to_csv(str(out_prefix) + "_finetuned_matrix.csv")
    delta_df.to_csv(str(out_prefix) + "_delta_matrix.csv")

    print(f"Wrote: {out_prefix}_pretrained_matrix.csv")
    print(f"Wrote: {out_prefix}_finetuned_matrix.csv")
    print(f"Wrote: {out_prefix}_delta_matrix.csv")


if __name__ == "__main__":
    main()
