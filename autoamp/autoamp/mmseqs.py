"""Module for clustering sequences using MMSeqs2."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from autoamp.finetune import Sequence
from autoamp.finetune import write_fasta


def mmseqs2_cluster(
    sequences: list[Sequence],
    min_seq_id: float = 0.5,
    cluster_coverage: float = 0.8,
    cov_mode: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Cluster sequences using MMSeqs2.

    Parameters
    ----------
    sequences : list[Sequence]
        List of sequences to cluster.
    min_seq_id : float, optional
        Minimum sequence identity, by default 0.5
    cluster_coverage : float, optional
        Minimum cluster coverage, by default 0.8
    cov_mode : int, optional
        Coverage mode, by default 1
    verbose : bool, optional
        Whether to print the mmseqs2 output, by default False

    Returns
    -------
    pd.DataFrame
        Dataframe containing the clusters.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # Write sequences to a temporary file
        input_file = Path(tmp) / 'input.fasta'
        write_fasta(sequences, input_file)

        # MMSeqs2 command
        command = (
            f'mmseqs easy-cluster {input_file} clusterRes tmp '
            f'--min-seq-id {min_seq_id} '
            f'-c {cluster_coverage} '
            f'--cov-mode {cov_mode}'
        ).split()

        # Run MMSeqs2
        if verbose:
            subprocess.run(command, check=True, cwd=tmp)
        else:
            with open(Path(tmp) / 'mmseqs2.log', 'w') as f:
                subprocess.run(
                    command, check=True, stdout=f, stderr=f, cwd=tmp
                )

        # Load the clustering result
        cluster_file = Path(tmp) / 'clusterRes_cluster.tsv'
        clusters = pd.read_csv(
            cluster_file,
            sep='\t',
            header=None,
            names=['ClusterID', 'SequenceID'],
        )

        # Group sequences by clusters
        groups = clusters.groupby('ClusterID')['SequenceID'].apply(list)

    return groups
