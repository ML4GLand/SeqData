from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
from .utils import (
    _tokenize,
    _sequencize,
    _token2one_hot,
    _one_hot2token,
    _pad_sequences,
)  # modified concise
from .utils import (
    _string_to_char_array,
    _one_hot_to_tokens,
    _char_array_to_string,
    _tokens_to_one_hot,
)  # dinuc_shuffle


# Vocabularies
DNA = ["A", "C", "G", "T"]
RNA = ["A", "C", "G", "U"]
COMPLEMENT_DNA = {"A": "T", "C": "G", "G": "C", "T": "A"}
COMPLEMENT_RNA = {"A": "U", "C": "G", "G": "C", "U": "A"}


def sanitize_seq(seq):
    """Capitalizes and removes whitespace for single seq."""
    return seq.strip().upper()


def sanitize_seqs(seqs):
    """Capitalizes and removes whitespace for a set of sequences."""
    return np.array([seq.strip().upper() for seq in seqs])


def reverse_complement_seq(seq, vocab="DNA"):
    """Reverse complement a single sequence."""
    if isinstance(seq, str):
        if vocab == "DNA":
            return "".join(COMPLEMENT_DNA.get(base, base) for base in reversed(seq))
        elif vocab == "RNA":
            return "".join(COMPLEMENT_RNA.get(base, base) for base in reversed(seq))
        else:
            raise ValueError("Invalid vocab, only DNA or RNA are currently supported")
    elif isinstance(seq, np.ndarray):
        return torch.from_numpy(np.flip(seq, axis=(0, 1)).copy()).numpy()


def reverse_complement_seqs(seqs, vocab="DNA", verbose=True):
    """Reverse complement set of sequences."""
    if isinstance(seqs[0], str):
        return np.array(
            [
                reverse_complement_seq(seq, vocab)
                for i, seq in tqdm(
                    enumerate(seqs),
                    total=len(seqs),
                    desc="Reverse complementing sequences",
                    disable=not verbose,
                )
            ]
        )
    elif isinstance(seqs[0], np.ndarray):
        return torch.from_numpy(np.flip(seqs, axis=(1, 2)).copy()).numpy()
