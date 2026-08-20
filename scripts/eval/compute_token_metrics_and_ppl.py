#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    records = []
    header = None
    seq_lines = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            records.append((header, "".join(seq_lines)))

    return records


def infer_base_model_from_adapter(adapter_dir):
    p = Path(adapter_dir) / "adapter_config.json"
    if not p.exists():
        return None

    with open(p) as f:
        cfg = json.load(f)

    return cfg.get("base_model_name_or_path")


def load_model_and_tokenizer(model_type, model_path, base_model=None, device="auto"):
    model_path = str(model_path)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    is_adapter = (Path(model_path) / "adapter_config.json").exists()

    if is_adapter:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("This model looks like a LoRA adapter, but peft is not installed.") from e

        if base_model is None:
            base_model = infer_base_model_from_adapter(model_path)

        if base_model is None:
            raise ValueError(
                "Model is a LoRA adapter but no --base_model was provided "
                "and adapter_config.json did not contain base_model_name_or_path."
            )

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        if model_type == "mlm":
            base = AutoModelForMaskedLM.from_pretrained(base_model, trust_remote_code=True)
        elif model_type == "causal":
            base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
        else:
            raise ValueError(model_type)

        model = PeftModel.from_pretrained(base, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if model_type == "mlm":
            model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
        elif model_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        else:
            raise ValueError(model_type)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    model.to(device)
    model.eval()

    return tokenizer, model, device


def aa_token_ids(tokenizer):
    aa_to_id = {}

    for aa in AA_LIST:
        ids = tokenizer(aa, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            aa_to_id[aa] = ids[0]

    if len(aa_to_id) < 20:
        missing = [aa for aa in AA_LIST if aa not in aa_to_id]
        raise ValueError(
            f"Tokenizer did not map all 20 amino acids to single tokens. Missing: {missing}"
        )

    return aa_to_id


def choose_sequence_format(tokenizer, seq, aa_id_set, mode):
    if mode == "raw":
        return seq

    if mode == "spaced":
        return " ".join(seq)

    raw_ids = tokenizer(seq, add_special_tokens=True)["input_ids"]
    spaced_ids = tokenizer(" ".join(seq), add_special_tokens=True)["input_ids"]

    raw_count = sum(1 for x in raw_ids if x in aa_id_set)
    spaced_count = sum(1 for x in spaced_ids if x in aa_id_set)

    # Choose whichever tokenization recovers more amino-acid residue tokens.
    if spaced_count > raw_count:
        return " ".join(seq)

    return seq


class MetricAccumulator:
    def __init__(self, label_ids, id_to_aa):
        self.label_ids = list(label_ids)
        self.label_set = set(label_ids)
        self.id_to_aa = dict(id_to_aa)

        self.tp = {x: 0 for x in self.label_ids}
        self.fp = {x: 0 for x in self.label_ids}
        self.fn = {x: 0 for x in self.label_ids}
        self.support = {x: 0 for x in self.label_ids}

        self.total = 0
        self.correct = 0
        self.non_aa_predictions = 0

        self.total_nll = 0.0
        self.total_nll_tokens = 0

        self.sequence_nlls = []
        self.sequence_ppls = []

    def update_predictions(self, y_true, y_pred):
        for y, p in zip(y_true, y_pred):
            y = int(y)
            p = int(p)

            if y not in self.label_set:
                continue

            self.total += 1
            self.support[y] += 1

            if p not in self.label_set:
                self.non_aa_predictions += 1

            if p == y:
                self.correct += 1
                self.tp[y] += 1
            else:
                self.fn[y] += 1
                if p in self.label_set:
                    self.fp[p] += 1

    def update_nll(self, nll_values, sequence_breakdown=None):
        vals = [float(x) for x in nll_values]
        self.total_nll += sum(vals)
        self.total_nll_tokens += len(vals)

        if sequence_breakdown is not None:
            for nll_sum, count in sequence_breakdown:
                if count > 0:
                    mean_nll = float(nll_sum) / int(count)
                    self.sequence_nlls.append(mean_nll)
                    self.sequence_ppls.append(math.exp(mean_nll))

    def summary(self):
        accuracy = self.correct / self.total if self.total else 0.0

        precisions = []
        recalls = []
        f1s = []
        per_class = []

        for label_id in self.label_ids:
            tp = self.tp[label_id]
            fp = self.fp[label_id]
            fn = self.fn[label_id]
            support = self.support[label_id]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            per_class.append({
                "amino_acid": self.id_to_aa[label_id],
                "token_id": label_id,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

        corpus_nll = self.total_nll / self.total_nll_tokens if self.total_nll_tokens else float("nan")
        corpus_ppl = math.exp(corpus_nll) if self.total_nll_tokens else float("nan")

        mean_sequence_nll = (
            sum(self.sequence_nlls) / len(self.sequence_nlls)
            if self.sequence_nlls
            else float("nan")
        )

        mean_sequence_ppl = (
            sum(self.sequence_ppls) / len(self.sequence_ppls)
            if self.sequence_ppls
            else float("nan")
        )

        return {
            "accuracy": accuracy,
            "macro_precision": sum(precisions) / len(precisions),
            "macro_recall": sum(recalls) / len(recalls),
            "macro_f1": sum(f1s) / len(f1s),
            "total_scored_tokens": self.total,
            "correct_tokens": self.correct,
            "non_aa_predictions": self.non_aa_predictions,
            "corpus_nll": corpus_nll,
            "corpus_pseudo_perplexity": corpus_ppl,
            "mean_sequence_nll": mean_sequence_nll,
            "mean_sequence_pseudo_perplexity": mean_sequence_ppl,
            "num_sequences_with_scored_tokens": len(self.sequence_ppls),
            "per_class": per_class,
        }


@torch.no_grad()
def score_mlm(records, tokenizer, model, device, aa_to_id, mask_batch_size, sequence_format):
    aa_id_set = set(aa_to_id.values())
    id_to_aa = {v: k for k, v in aa_to_id.items()}
    label_ids = [aa_to_id[aa] for aa in AA_LIST]

    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer has no mask token; MLM scoring requires mask_token_id.")

    acc = MetricAccumulator(label_ids=label_ids, id_to_aa=id_to_aa)

    for idx, (header, seq) in enumerate(records, start=1):
        formatted = choose_sequence_format(tokenizer, seq, aa_id_set, sequence_format)

        enc = tokenizer(formatted, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        positions = [
            i for i, tok in enumerate(input_ids.tolist())
            if tok in aa_id_set
        ]

        if not positions:
            continue

        seq_nll_sum = 0.0
        seq_count = 0

        for start in range(0, len(positions), mask_batch_size):
            batch_positions = positions[start:start + mask_batch_size]
            bsz = len(batch_positions)

            batch_ids = input_ids.unsqueeze(0).repeat(bsz, 1)
            batch_attn = attention_mask.unsqueeze(0).repeat(bsz, 1)

            for row, pos in enumerate(batch_positions):
                batch_ids[row, pos] = tokenizer.mask_token_id

            batch_ids = batch_ids.to(device)
            batch_attn = batch_attn.to(device)

            out = model(input_ids=batch_ids, attention_mask=batch_attn)
            logits = out.logits

            pos_tensor = torch.tensor(batch_positions, dtype=torch.long, device=device)
            row_tensor = torch.arange(bsz, dtype=torch.long, device=device)

            site_logits = logits[row_tensor, pos_tensor, :]
            log_probs = F.log_softmax(site_logits, dim=-1)

            targets = input_ids[batch_positions].to(device)
            preds = torch.argmax(site_logits, dim=-1)

            nll = -log_probs[row_tensor, targets]

            nll_cpu = nll.detach().float().cpu().tolist()
            targets_cpu = targets.detach().cpu().tolist()
            preds_cpu = preds.detach().cpu().tolist()

            acc.update_predictions(targets_cpu, preds_cpu)
            acc.update_nll(nll_cpu)

            seq_nll_sum += sum(nll_cpu)
            seq_count += len(nll_cpu)

        acc.update_nll([], sequence_breakdown=[(seq_nll_sum, seq_count)])

        if idx % 50 == 0:
            print(f"Scored {idx}/{len(records)} sequences", flush=True)

    return acc.summary()


@torch.no_grad()
def score_causal(records, tokenizer, model, device, aa_to_id, batch_size, sequence_format):
    aa_id_set = set(aa_to_id.values())
    id_to_aa = {v: k for k, v in aa_to_id.items()}
    label_ids = [aa_to_id[aa] for aa in AA_LIST]

    acc = MetricAccumulator(label_ids=label_ids, id_to_aa=id_to_aa)

    aa_mask = torch.zeros(len(tokenizer), dtype=torch.bool)
    for tok_id in aa_id_set:
        if tok_id < len(aa_mask):
            aa_mask[tok_id] = True
    aa_mask = aa_mask.to(device)

    for start in range(0, len(records), batch_size):
        batch_records = records[start:start + batch_size]
        seqs = [
            choose_sequence_format(tokenizer, seq, aa_id_set, sequence_format)
            for _, seq in batch_records
        ]

        enc = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=True,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits

        # Predict token t using logits at t-1.
        pred_logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]

        valid = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
        valid = valid & aa_mask[targets]

        log_probs = F.log_softmax(pred_logits, dim=-1)
        preds = torch.argmax(pred_logits, dim=-1)

        nll_all = -torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        for row in range(input_ids.shape[0]):
            mask = valid[row]
            if not torch.any(mask):
                continue

            row_targets = targets[row][mask]
            row_preds = preds[row][mask]
            row_nll = nll_all[row][mask]

            nll_cpu = row_nll.detach().float().cpu().tolist()
            targets_cpu = row_targets.detach().cpu().tolist()
            preds_cpu = row_preds.detach().cpu().tolist()

            acc.update_predictions(targets_cpu, preds_cpu)
            acc.update_nll(nll_cpu)
            acc.update_nll([], sequence_breakdown=[(sum(nll_cpu), len(nll_cpu))])

        done = min(start + batch_size, len(records))
        if done % 50 == 0 or done == len(records):
            print(f"Scored {done}/{len(records)} sequences", flush=True)

    return acc.summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["mlm", "causal"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--mask_batch_size",
        type=int,
        default=32,
        help="For MLM scoring, number of masked sequence variants per forward pass.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--sequence_format",
        default="auto",
        choices=["auto", "raw", "spaced"],
        help="Use raw sequences, space-separated amino acids, or auto-detect.",
    )
    parser.add_argument("--max_sequences", type=int, default=None)
    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    records = read_fasta(args.fasta)
    if args.max_sequences is not None:
        records = records[:args.max_sequences]

    print(f"Loaded {len(records)} sequences from {args.fasta}", flush=True)

    tokenizer, model, device = load_model_and_tokenizer(
        model_type=args.model_type,
        model_path=args.model,
        base_model=args.base_model,
        device=args.device,
    )

    aa_to_id = aa_token_ids(tokenizer)

    print(f"Device: {device}", flush=True)
    print(f"Model type: {args.model_type}", flush=True)
    print(f"Model label: {args.model_label}", flush=True)
    print(f"AA token IDs: {aa_to_id}", flush=True)

    if args.model_type == "mlm":
        metrics = score_mlm(
            records=records,
            tokenizer=tokenizer,
            model=model,
            device=device,
            aa_to_id=aa_to_id,
            mask_batch_size=args.mask_batch_size,
            sequence_format=args.sequence_format,
        )
    else:
        metrics = score_causal(
            records=records,
            tokenizer=tokenizer,
            model=model,
            device=device,
            aa_to_id=aa_to_id,
            batch_size=args.batch_size,
            sequence_format=args.sequence_format,
        )

    metrics.update({
        "model_label": args.model_label,
        "model_type": args.model_type,
        "model": args.model,
        "base_model": args.base_model,
        "fasta": args.fasta,
        "num_input_sequences": len(records),
        "batch_size": args.batch_size,
        "mask_batch_size": args.mask_batch_size,
        "sequence_format": args.sequence_format,
    })

    json_out = out_prefix.with_suffix(".json")
    class_out = out_prefix.with_name(out_prefix.name + "_per_class.csv")

    with open(json_out, "w") as f:
        json.dump(metrics, f, indent=2)

    import csv
    with open(class_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "amino_acid",
                "token_id",
                "support",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics["per_class"])

    print()
    print("==== Metrics ====")
    for key in [
        "accuracy",
        "macro_recall",
        "macro_precision",
        "macro_f1",
        "corpus_pseudo_perplexity",
        "mean_sequence_pseudo_perplexity",
        "total_scored_tokens",
        "num_sequences_with_scored_tokens",
    ]:
        print(f"{key}: {metrics[key]}")

    print()
    print(f"Wrote: {json_out}")
    print(f"Wrote: {class_out}")


if __name__ == "__main__":
    main()
