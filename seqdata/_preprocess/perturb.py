


def dinuc_shuffle_seq(seq, num_shufs=None, rng=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.

    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D np array, then the
    result is an N x L x D np array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).

    Parameters
    ----------
    seq : str
        The sequence to shuffle.
    num_shufs : int, optional
        The number of shuffles to create. If None, only one shuffle is created.
    rng : np.random.RandomState, optional
        The random number generator to use. If None, a new one is created.

    Returns
    -------
    list of str or np.array
        The shuffled sequences.

    Note
    ----
    This function comes from DeepLIFT's dinuc_shuffle.py.
    """
    if type(seq) is str or type(seq) is np.str_:
        arr = _string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = _one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")
    if not rng:
        rng = np.random.RandomState()

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    if type(seq) is str or type(seq) is np.str_:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim), dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str or type(seq) is np.str_:
            all_results.append(_char_array_to_string(chars[result]))
        else:
            all_results[i] = _tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]


def dinuc_shuffle_seqs(seqs, num_shufs=None, rng=None):
    """
    Shuffle the sequences in `seqs` in the same way as `dinuc_shuffle_seq`.
    If `num_shufs` is not specified, then the first dimension of N will not be
    present (i.e. a single string will be returned, or an L x D array).

    Parameters
    ----------
    seqs : np.ndarray
        Array of sequences to shuffle
    num_shufs : int, optional
        Number of shuffles to create, by default None
    rng : np.random.RandomState, optional
        Random state to use for shuffling, by default None

    Returns
    -------
    np.ndarray
        Array of shuffled sequences

    Note
    -------
    This is taken from DeepLIFT
    """
    if not rng:
        rng = np.random.RandomState()

    if type(seqs) is str or type(seqs) is np.str_:
        seqs = [seqs]

    all_results = []
    for i in range(len(seqs)):
        all_results.append(dinuc_shuffle_seq(seqs[i], num_shufs=num_shufs, rng=rng))
    return np.array(all_results)


def perturb_seq(X_0, vocab_len=4):
    """
    Produce all edit-distance-one pertuabtions for a single of sequences.

    Note
    ----
    This function is modified from the Yuzu package
    """
    import warnings

    if not isinstance(X_0, np.ndarray):
        raise ValueError("X_0 must be of type np.ndarray, not {}".format(type(X_0)))

    if len(X_0.shape) != 2:
        raise ValueError("X_0 must have two dimensions: (n_choices, seq_len).")

    if X_0.shape[0] != 4:
        warnings.warn(
            "X_0 has {} choices, but should have 4. Transposing".format(X_0.shape[1])
        )
        X_0 = X_0.transpose()

    n_choices, seq_len = X_0.shape
    idx = X_0.argmax(axis=0)
    X_0 = torch.from_numpy(X_0)

    n = seq_len * (n_choices - 1)
    X = torch.tile(X_0, (n, 1))
    X = X.reshape(n, n_choices, seq_len)
    for k in range(1, n_choices):
        i = np.arange(seq_len) * (n_choices - 1) + (k - 1)
        X[i, idx, np.arange(seq_len)] = 0
        X[i, (idx + k) % n_choices, np.arange(seq_len)] = 1

    return X.numpy()


def perturb_seqs(X_0, vocab_len=4):
    """
    Produce all edit-distance-one pertuabtions for a set of sequences.

    This function will take in a single one-hot encoded sequence of length N
    and return a batch of N*(n_choices-1) sequences, each containing a single
    perturbation from the given sequence.

    Parameters
    ----------
    X_0: np.ndarray, shape=(n_seqs, n_choices, seq_len)
        A one-hot encoded sequence to generate all potential perturbations.

    Returns
    -------
    X: torch.Tensor, shape=(n_seqs, (n_choices-1)*seq_len, n_choices, seq_len)
        Each single-position perturbation of seq.

    Note
    ----
    This function is modified from the Yuzu package.
    """
    import warnings

    if not isinstance(X_0, np.ndarray):
        raise ValueError("X_0 must be of type np.ndarray, not {}".format(type(X_0)))

    if len(X_0.shape) != 3:
        raise ValueError(
            "X_0 must have three dimensions: (n_seqs, n_choices, seq_len)."
        )
    if X_0.shape[1] != 4:
        warnings.warn(
            "X_0 has {} choices, but should have 4. Transposing".format(X_0.shape[1])
        )
        X_0 = X_0.transpose(0, 2, 1)

    n_seqs, n_choices, seq_len = X_0.shape
    idxs = X_0.argmax(axis=1)

    X_0 = torch.from_numpy(X_0)

    n = seq_len * (n_choices - 1)
    X = torch.tile(X_0, (n, 1, 1))

    X = X.reshape(n, n_seqs, n_choices, seq_len).permute(1, 0, 2, 3)

    for i in range(n_seqs):
        for k in range(1, n_choices):
            idx = np.arange(seq_len) * (n_choices - 1) + (k - 1)

            X[i, idx, idxs[i], np.arange(seq_len)] = 0
            X[i, idx, (idxs[i] + k) % n_choices, np.arange(seq_len)] = 1

    return X


def feature_implant_seq(
    seq, feature, position, vocab="DNA", encoding="str", onehot=False
):
    """
    Insert a feature at a given position in a single sequence.
    """
    if encoding == "str":
        return seq[:position] + feature + seq[position + len(feature) :]
    elif encoding == "onehot":
        if onehot:
            feature = _token2one_hot(feature.argmax(axis=1), vocab=vocab, fill_value=0)
        if feature.shape[0] != seq.shape[0]:
            feature = feature.transpose()
        return np.concatenate(
            (seq[:, :position], feature, seq[:, position + feature.shape[-1] :]), axis=1
        )
    else:
        raise ValueError("Encoding not recognized.")


def feature_implant_across_seq(seq, feature, **kwargs):
    """
    Insert a feature at every position for a single sequence.
    """
    if isinstance(seq, str):
        assert isinstance(feature, str)
        seq_len = len(seq)
        feature_len = len(feature)
    elif isinstance(seq, np.ndarray):
        assert isinstance(feature, np.ndarray)
        seq_len = seq.shape[-1]
        if feature.shape[0] != seq.shape[0]:
            feature_len = feature.shape[0]
        else:
            feature_len = feature.shape[-1]
    implanted_seqs = []
    for pos in range(seq_len - feature_len + 1):
        seq_implanted = feature_implant_seq(seq, feature, pos, **kwargs)
        implanted_seqs.append(seq_implanted)
    return np.array(implanted_seqs)