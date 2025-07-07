"""CLI for autoamp."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def random_split(
    input_fasta: Path = typer.Option(  # noqa: B008
        ...,
        '--input_fasta',
        '-i',
        help='The fasta file to split.',
    ),
    output_dir: Path = typer.Option(  # noqa: B008
        ...,
        '--output_dir',
        '-o',
        help='The directory to save the split fasta files.',
    ),
    split: float = typer.Option(
        0.8,
        '--split',
        '-s',
        help='The fraction of sequences to put in the first file.',
    ),
    seed: int = typer.Option(
        0,
        '--seed',
        '-r',
        help='The seed for the random number generator.',
    ),
) -> None:
    """Randomly split a fasta file into a train and validation set."""
    from autoamp.finetune import random_split_fasta

    random_split_fasta(input_fasta, output_dir, split, seed)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == '__main__':
    main()
