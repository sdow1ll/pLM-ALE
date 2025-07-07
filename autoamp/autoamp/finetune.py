"""Finetune the ESM model."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from argparse import ArgumentParser
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers
import wandb
import yaml
from torch.utils.data import Dataset
from transformers import EsmForMaskedLM
from transformers import EsmTokenizer
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint

PathLike = str | Path


@dataclass
class Sequence:
    """Store a biological sequence and its description tag."""

    sequence: str
    """Biological sequence (Nucleotide sequence)."""
    tag: str
    """Sequence description tag."""

    def __hash__(self) -> int:
        """Hash the sequence and tag."""
        return hash((self.sequence, self.tag))


def read_fasta(fasta_file: PathLike) -> list[Sequence]:
    """Read fasta file sequences and description tags into dataclass."""
    text = Path(fasta_file).read_text()
    pattern = re.compile('^>', re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace('\n', '')
        for seq in non_parsed_seqs
        for line in seq.split('\n', 1)
    ]

    return [
        Sequence(sequence=seq, tag=tag)
        for seq, tag in zip(lines[1::2], lines[::2])
    ]


def write_fasta(
    sequences: Sequence | list[Sequence],
    fasta_file: PathLike,
    mode: str = 'w',
) -> None:
    """Write or append sequences to a fasta file."""
    seqs = [sequences] if isinstance(sequences, Sequence) else sequences
    with open(fasta_file, mode) as f:
        f.write('\n'.join(f'>{seq.tag}\n{seq.sequence}' for seq in seqs))


def random_split_fasta(
    input_fasta: PathLike,
    output_dir: PathLike,
    split: float = 0.8,
    seed: int = 0,
) -> None:
    """Randomly split a fasta file into train and validation fasta file."""
    # Read the input file
    sequences = read_fasta(input_fasta)

    # Shuffle the sequences
    random.seed(seed)
    random.shuffle(sequences)

    # Create the output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Write the train and validation fasta files
    split_idx = int(len(sequences) * split)
    write_fasta(sequences[:split_idx], output_dir / 'train.fasta')
    write_fasta(sequences[split_idx:], output_dir / 'valid.fasta')

    # Copy the original fasta file to the output directory for reference
    shutil.copy(input_fasta, output_dir)

    # Log JSON metadata on the split
    metadata = {
        'input_fasta': str(Path(input_fasta).resolve()),
        'output_dir': str(output_dir.resolve()),
        'split': split,
        'seed': seed,
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Write the metadata to a JSON file
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


class SequenceDataset(Dataset):
    """Dataset for sequences."""

    def __init__(self, sequences: list[str]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        """Return the idx'th sequence."""
        # Get the idx'th sequence
        return self.sequences[idx]


class DataCollator(transformers.DataCollatorForLanguageModeling):
    """Data collator for language modeling with ESM."""

    def __init__(self, train_mode: bool = False, **kwargs: Any) -> None:
        self.train_mode = train_mode
        super().__init__(**kwargs)

    def tokenize(self, sequences: list[str]) -> transformers.BatchEncoding:
        """Tokenize the sequences and return a BatchEncoding."""
        return self.tokenizer(
            sequences,
            return_tensors='pt',
            truncation=True,
            padding=True,
            return_special_tokens_mask=self.train_mode and self.mlm,
        )

    def torch_call(self, examples: list[str]) -> transformers.BatchEncoding:
        """Tokenize the batch and prepare the input and labels."""
        # First, tokenize the batch
        batch = self.tokenize(examples)
        #print(f'{batch}')
        # We only need to mask tokens if we are doing inference
        if not self.train_mode:
            return batch

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop('special_tokens_mask', None)

        if self.mlm:
            batch['input_ids'], batch['labels'] = self.torch_mask_tokens(
                batch['input_ids'], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch['input_ids'].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch['labels'] = labels
        return batch


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """TrainingArguments for configuring the Hugging Face Trainer.

    Here we provide some sensible defaults for the arguments for our use case.
    """

    output_dir: str = field(
        default='test_run',
        metadata={
            'help': 'The output directory where the model predictions and '
            'checkpoints will be written.'
        },
    )
    per_device_train_batch_size: int = field(
        default=64,
        metadata={'help': 'Batch size per GPU/TPU core/CPU for training.'},
    )
    per_device_eval_batch_size: int = field(
        default=128,
        metadata={'help': 'Batch size per GPU/TPU core/CPU for evaluation.'},
    )
    num_train_epochs: float = field(
        default=20,
        metadata={'help': 'Total number of training epochs to perform.'},
    )
    learning_rate: float = field(
        default=4e-4,
        metadata={'help': 'The initial learning rate for Adam.'},
    )
    warmup_steps: int = field(
        default=1_000,
        metadata={'help': 'Linear warmup over `warmup_steps`.'},
    )
    lr_scheduler_type: str = field(
        default='cosine',
        metadata={'help': 'The scheduler type to use.'},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={'help': 'The weight decay to apply.'},
    )
    eval_steps: int = field(
        default=500,
        metadata={
            'help': 'Number of steps between evaluations. If `eval_steps` '
            'is modified, update `logging_steps` and `save_steps` to the same '
            'value.'
        },
    )
    save_total_limit: int = field(
        default=1,
        metadata={'help': 'Total number of checkpoints to save.'},
    )
    save_strategy: str = field(
        default='steps',
        metadata={'help': 'Strategy for saving checkpoints.'},
    )
    evaluation_strategy: str = field(
        default='steps',
        metadata={'help': 'Strategy for evaluating.'},
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={
            'help': 'Whether to load the best model at the end of training. '
            'When `save_total_limit` is set to 1, will save the best model as '
            'well as the last model if the last model is worse (eval_loss) '
            'than the best model.'
        },
    )
    fp16: bool = field(
        default=True,
        metadata={'help': 'Whether to use 16-bit (mixed) precision training.'},
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={'help': 'Number of subprocesses to use for data loading.'},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            'help': 'This skips underlying logic in Trainer which modifies '
            'the data_collator (do not change).'
        },
    )


@dataclass
class TrainingConfig:
    """Configuration for fine tuning the ESM model."""

    train_path: str = field(
        metadata={'help': 'Path to training data.'},
    )
    eval_path: str = field(
        metadata={'help': 'Path to validation data.'},
    )
    training_args: TrainingArguments = field(
        default_factory=TrainingArguments,
        metadata={
            'help': 'Hugging face arguments for training the model '
            '(see transformers.TrainingArguments).'
        },
    )
    base_model: str = field(
        default='facebook/esm2_t6_8M_UR50D',
        metadata={'help': 'Base model to use for training.'},
    )
    wandb_project: str = field(
        default='',
        metadata={
            'help': 'Wandb project name (By default, set to empty string'
            ' to turn off wandb).'
        },
    )

    def __post_init__(self) -> None:
        """Initialize the training arguments and log the config."""
        # Populate the training arguments
        self.training_args = TrainingArguments(**self.training_args)

        # Set the output directory
        output_dir = Path(self.training_args.output_dir)

        # Create the output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True, parents=True)

        # wandb needs to be initialized once on all node ranks
        if self.wandb_project and self.training_args.local_process_index == 0:
            os.environ['WANDB_PROJECT'] = self.wandb_project
            # Assign the same group name as the output directory
            # so that multi-node runs are grouped together
            wandb.init(dir=output_dir, group=output_dir.name)
            wandb.config.update(
                {'train_config': asdict(self)}, allow_val_change=True
            )

        # Set the report_to argument based on the wandb project
        self.training_args.report_to = ['wandb' if self.wandb_project else '']

        # Log the config to a yaml file
        with open(output_dir / 'train_config.yaml', 'w') as fp:
            yaml.dump(asdict(self), fp)


class ClearEvalMemoryTrainer(Trainer):
    """Trainer that clears the cuda cache before each evaluation.

    Note: reduces OOMs for some models.
    """

    def _clear_cuda_cache(self) -> None:
        import gc

        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.clear_autocast_cache()

    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """Clear the cuda cache before evaluation."""
        self._clear_cuda_cache()
        return super().evaluate(*args, **kwargs)


def main() -> None:
    """Finetune the ESM model."""
    # Parse a yaml file to get the training config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as fp:
        config = TrainingConfig(**yaml.safe_load(fp))

    # Load the model from the model name
    model = EsmForMaskedLM.from_pretrained(config.base_model)

    # Load the tokenizer
    tokenizer = EsmTokenizer.from_pretrained(config.base_model)

    # Set the model max length for proper truncation
    tokenizer.model_max_length = model.config.max_position_embeddings

    # Read the fasta file into a list of sequences
    train_sequences = [seq.sequence for seq in read_fasta(config.train_path)]
    eval_sequences = [seq.sequence for seq in read_fasta(config.eval_path)]

    # Construct the train and validation datasets
    train_dataset = SequenceDataset(train_sequences)
    eval_dataset = SequenceDataset(eval_sequences)

    # Create the data collator
    data_collator = DataCollator(
        train_mode=True,
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        pad_to_multiple_of=8 if config.training_args.fp16 else None,
    )

    # Create the trainer
    trainer = ClearEvalMemoryTrainer(
        model=model,
        args=config.training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Attempt to load a checkpoint
    checkpoint = get_last_checkpoint(config.training_args.output_dir)
    if checkpoint is not None:
        print('Training from checkpoint:', checkpoint)

    # Train the model
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Saves the tokenizer too for easy upload
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()


if __name__ == '__main__':
    main()
