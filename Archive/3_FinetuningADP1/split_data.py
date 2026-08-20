#!/usr/bin/env python3
import os
import argparse
import random
from Bio import SeqIO

def main():
    parser = argparse.ArgumentParser(
        description=("Split a FASTA file into train, validation, and test sets "
                     "and save them into separate folders.")
    )
    parser.add_argument('fasta', help='Path to the input FASTA file (e.g. dgoa_mutants.fasta)')
    parser.add_argument('--train', type=float, default=0.8,
                        help='Proportion for training set (default: 0.8)')
    parser.add_argument('--val', type=float, default=0.1,
                        help='Proportion for validation set (default: 0.1)')
    parser.add_argument('--test', type=float, default=0.1,
                        help='Proportion for test set (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    # Check that the proportions add up to 1.0 (or very close)
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("The sum of train, val, and test proportions must equal 1.0")

    # Read in all sequences from the FASTA file
    sequences = list(SeqIO.parse(args.fasta, "fasta"))
    if not sequences:
        raise ValueError("No sequences found in the input FASTA file.")

    # Shuffle the sequences for randomized splitting
    random.seed(args.seed)
    random.shuffle(sequences)
    total_sequences = len(sequences)

    # Calculate the indices for splitting
    train_end = int(total_sequences * args.train)
    val_end = train_end + int(total_sequences * args.val)

    train_set = sequences[:train_end]
    val_set = sequences[train_end:val_end]
    test_set = sequences[val_end:]

    # Create a mapping for the split names and sets
    splits = {
        'train': train_set,
        'validation': val_set,
        'test': test_set
    }

    # Determine the base name of the input file (without extension)
    base_name = os.path.splitext(os.path.basename(args.fasta))[0]

    # Create directories and write FASTA files for each split
    for split_name, seqs in splits.items():
        # Create directory if it does not exist
        os.makedirs(split_name, exist_ok=True)
        output_filename = os.path.join(split_name, f"{base_name}_{split_name}.fasta")
        with open(output_filename, "w") as output_handle:
            SeqIO.write(seqs, output_handle, "fasta")
        print(f"Wrote {len(seqs)} sequences to {output_filename}")

if __name__ == "__main__":
    main()
