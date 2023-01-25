def ascii_encode_seq(seq, pad=0):
    """
    Converts a string of characters to a NumPy array of byte-long ASCII codes.
    """
    encode_seq = np.array([ord(letter) for letter in seq], dtype=int)
    if pad > 0:
        encode_seq = np.pad(
            encode_seq, pad_width=(0, pad), mode="constant", constant_values=36
        )
    return encode_seq


def ascii_encode_seqs(seqs, pad=0):
    """
    Converts a set of sequences to a NumPy array of byte-long ASCII codes.
    """
    encode_seqs = np.array(
        [ascii_encode_seq(seq, pad=pad) for seq in seqs], dtype=int
    )
    return encode_seqs


def ascii_decode_seq(seq):
    """
    Converts a NumPy array of byte-long ASCII codes to a string of characters.
    """
    return "".join([chr(int(letter)) for letter in seq]).replace("$", "")


def ascii_decode_seqs(seqs):
    """Convert a set of one-hot encoded arrays back to strings"""
    return np.array(
        [ascii_decode_seq(seq) for seq in seqs], dtype=object
    )


def ohe_seq(seq, vocab="DNA", neutral_vocab="N", fill_value=0):
    """Convert a sequence into one-hot-encoded array."""
    seq = seq.strip().upper()
    return _token2one_hot(
        _tokenize(seq, vocab, neutral_vocab), vocab, fill_value=fill_value
    )


def ohe_seqs(
    seqs,
    vocab="DNA",
    neutral_vocab="N",
    maxlen=None,
    pad=True,
    pad_value="N",
    fill_value=None,
    seq_align="start",
    verbose=True,
):
    """Convert a set of sequences into one-hot-encoded array."""
    if isinstance(neutral_vocab, str):
        neutral_vocab = [neutral_vocab]
    if isinstance(seqs, str):
        raise ValueError("seq_vec should be an iterable not a string itself")
    assert len(vocab[0]) == len(pad_value)
    assert pad_value in neutral_vocab
    if pad:
        seqs_vec = _pad_sequences(seqs, maxlen=maxlen, align=seq_align, value=pad_value)
    arr_list = [
        ohe_seq(
            seq=seqs_vec[i],
            vocab=vocab,
            neutral_vocab=neutral_vocab,
            fill_value=fill_value,
        )
        for i in tqdm(
            range(len(seqs_vec)),
            total=len(seqs_vec),
            desc="One-hot encoding sequences",
            disable=not verbose,
        )
    ]
    if pad:
        return np.stack(arr_list)
    else:
        return np.array(arr_list, dtype=object)


def decode_seq(arr, vocab="DNA", neutral_value=-1, neutral_char="N"):
    """Convert a single one-hot encoded array back to string"""
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    return _sequencize(
        tvec=_one_hot2token(arr, neutral_value),
        vocab=vocab,
        neutral_value=neutral_value,
        neutral_char=neutral_char,
    )


def decode_seqs(arr, vocab="DNA", neutral_char="N", neutral_value=-1, verbose=True):
    """Convert a one-hot encoded array back to set of sequences"""
    arr_list = [
        decode_seq(
            arr=arr[i],
            vocab=vocab,
            neutral_value=neutral_value,
            neutral_char=neutral_char,
        )
        for i in tqdm(
            range(len(arr)),
            total=len(arr),
            desc="Decoding sequences",
            disable=not verbose,
        )
    ]
    return np.array(arr_list)