"""Filtering functions for AMP sequences.

These follow thermofisher's guidelines for DMSO solvent.
"""

from __future__ import annotations

from typing import Protocol

from autoamp.finetune import Sequence


class Filter(Protocol):
    """Protocol for filtering sequences."""

    def apply(self, seqs: list[Sequence]) -> list[Sequence]:
        """Filter sequences based on the implemented criteria."""
        ...


class FilterChain(Filter):
    """Chain multiple filters together."""

    def __init__(self, filters: list[Filter], verbose: bool = False) -> None:
        self.filters = filters
        self.verbose = verbose

    def apply(self, seqs: list[Sequence]) -> list[Sequence]:
        """Filter sequences based on the implemented criteria."""
        # Print initial statistics
        if self.verbose:
            print(f'Initial number of sequences: {len(seqs)}')

        # Keep track of the original number of sequences
        starting_num = len(seqs)

        # Apply each filter in the chain
        for f in self.filters:
            # Keep track of the number of sequences before applying the filter
            original_num = len(seqs)

            # Filter the sequences
            seqs = f.apply(seqs)

            # Print statistics for each filter
            if self.verbose:
                print(
                    f'{f.__class__.__name__} filtered '
                    f'{original_num - len(seqs)}'
                )

        # Print final statistics
        if self.verbose:
            print(
                f'Total filtered: {starting_num - len(seqs)} '
                f'or {len(seqs) / starting_num * 100:.2f}%'
            )
            print(f'Final number of sequences: {len(seqs)}')

        return seqs


class LengthFilter(Filter):
    """Filter sequences based on length."""

    def __init__(self, threshold: int = 49) -> None:
        self.threshold = threshold

    def apply(self, seqs: list[Sequence]) -> list[Sequence]:
        """Filter sequences based on length."""
        return [s for s in seqs if len(s.sequence) <= self.threshold]


class DAminoAcidsFilter(Filter):
    """Filter sequences to exclude D-amino acids."""

    def apply(self, seqs: list[Sequence]) -> list[Sequence]:
        """Filter sequences to exclude D-amino acids."""
        # D-amino acids are lowercase
        return [s for s in seqs if s.sequence.isupper()]


class UnknownAminoAcidsFilter(Filter):
    """Filter sequences to exclude unknown amino acids."""

    def apply(self, seqs: list[Sequence]) -> list[Sequence]:
        """Filter sequences to exclude unknown amino acids."""
        # Unknown amino acids are represented by 'X'
        return [s for s in seqs if 'X' not in s.sequence]


def percent_in_group(seq: str, group: set[str]) -> float:
    """Calculate the percentage of residues in a given group."""
    return sum(1 for aa in seq if aa in group) / len(seq)


class ChargeFilter(Filter):
    """Filter sequences based on charge."""

    def __init__(self, threshold: float = 0.25) -> None:
        """Initialize the charge filter.

        Parameters
        ----------
        threshold : float, optional
            Minimum percentage of charged residues required, by default 0.25
        """
        self.threshold = threshold
        self.charged = set('DEHKR')

    def _apply(self, seq: str) -> bool:
        """Filter a single sequence based on charge."""
        return percent_in_group(seq, self.charged) > self.threshold

    def apply(self, seqs: list[Sequence]) -> list[Sequence]:
        """Filter sequences based on charge."""
        return [s for s in seqs if self._apply(s.sequence)]


class HydrophobicFilter(Filter):
    """Filter sequences based on hydrophobicity.

    Note: this helps make more sequence soluble.
    """

    def __init__(self, threshold: float = 0.50) -> None:
        """Initialize the hydrophobic filter.

        Parameters
        ----------
        threshold : float, optional
            Maximum percentage of hydrophobic residues allowed, by default 0.25
        """
        self.threshold = threshold
        self.hydrophobic = set('AILMFWV')

    def _apply(self, seq: str) -> bool:
        """Filter a single sequence based on hydrophobicity."""
        return percent_in_group(seq, self.hydrophobic) <= self.threshold

    def apply(self, seqs: list[Sequence]) -> list[Sequence]:
        """Filter sequences based on hydrophobicity."""
        return [s for s in seqs if self._apply(s.sequence)]


class PHAdjustmentFilter(Filter):
    """Filter sequences that require pH adjustment."""

    def __init__(self, threshold: float = 0.75) -> None:
        """Initialize the pH adjustment filter.

        Parameters
        ----------
        threshold : float, optional
            Maximum percentage of special group residues allowed,
            by default 0.75
        """
        self.threshold = threshold
        self.special_group = set('DEHKRNQRSTY')

    def _apply(self, seq: str) -> bool:
        """Filter a single sequence that require pH adjustment."""
        return percent_in_group(seq, self.special_group) <= self.threshold

    def apply(self, seqs: list[Sequence]) -> list[Sequence]:
        """Filter sequences that require pH adjustment."""
        return [s for s in seqs if self._apply(s.sequence)]


class NTerminalFilter(Filter):
    """Filter sequences based on the N-terminal residue."""

    def __init__(self) -> None:
        """Initialize the N-terminal filter."""
        self.disallowed = set('NQ')

    def apply(self, seqs: list[Sequence]) -> list[Sequence]:
        """Filter sequences based on the N-terminal residue."""
        return [s for s in seqs if s.sequence[0] not in 'NQ']


class DisallowedPairsFilter(Filter):
    """Filter sequences based on disallowed pairs of amino acids."""

    def __init__(self) -> None:
        """Initialize the disallowed pairs filter."""
        self.disallowed_pairs = {'D': set('GPS')}

    def _apply(self, seq: str) -> bool:
        """Filter a single sequence."""
        for i in range(len(seq) - 1):
            if seq[i] in self.disallowed_pairs:
                if seq[i + 1] in self.disallowed_pairs[seq[i]]:
                    return False

        return True

    def apply(self, seqs: list[Sequence]) -> list[Sequence]:
        """Filter sequences based on disallowed pairs of amino acids."""
        return [s for s in seqs if self._apply(s.sequence)]


class BetaSheetFilter(Filter):
    """Filter sequences leading to excessive beta sheet formation."""

    def __init__(self) -> None:
        """Initialize the beta sheet filter."""
        self.beta_sheet_prone = 'QILFPTYV'

    def _apply(self, seq: str) -> bool:
        """Filter a single sequence."""
        for i in range(len(seq) - 1):
            if (
                seq[i] in self.beta_sheet_prone
                and seq[i + 1] in self.beta_sheet_prone
            ):
                return False

        return True

    def apply(self, seqs: list[Sequence]) -> list[Sequence]:
        """Filter sequences based on beta sheet formation."""
        return [s for s in seqs if self._apply(s.sequence)]


class DuplicateFilter(Filter):
    """Filter sequences based on duplicates."""

    def apply(self, seqs: list[Sequence]) -> list[Sequence]:
        """Filter sequences based on duplicates."""
        seen = set()
        keep = []
        for s in seqs:
            # Importantly, we use the raw string sequence as a key
            # rather then the Sequence object itself which includes
            # additional tag metadata in the hash.
            if s.sequence not in seen:
                seen.add(s.sequence)
                keep.append(s)
        return keep


if __name__ == '__main__':
    from autoamp.finetune import read_fasta

    seq_path = '/Users/abrace/src/autoamp/unique_sequences.fasta'
    seqs = read_fasta(seq_path)

    # Setup the filters
    filters = [
        LengthFilter(threshold=49),
        DAminoAcidsFilter(),
        UnknownAminoAcidsFilter(),
        ChargeFilter(threshold=0.25),
        HydrophobicFilter(threshold=0.25),
        PHAdjustmentFilter(threshold=0.75),
        NTerminalFilter(),
        DisallowedPairsFilter(),
        BetaSheetFilter(),
    ]

    # Chain the filters
    chain = FilterChain(filters=filters, verbose=True)

    # Filter the sequences
    filtered_seqs = chain.apply(seqs)

    # Outputs:
    #  Initial number of sequences: 20296
    #  LengthFilter filtered 833
    #  DAminoAcidsFilter filtered 2265
    #  UnknownAminoAcidsFilter filtered 1580
    #  ChargeFilter filtered 6020
    #  HydrophobicFilter filtered 7805
    #  PHAdjustmentFilter filtered 370
    #  NTerminalFilter filtered 54
    #  DisallowedPairsFilter filtered 99
    #  BetaSheetFilter filtered 877
    #  Total filtered: 19903 or 1.94%
    #  Final number of sequences: 393
