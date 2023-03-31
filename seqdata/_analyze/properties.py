import numpy as np


def seq_len(seq, ohe=False):
    if ohe:
        return seq.shape[1]
    else:
        return len(seq)


def seq_lens(seqs, ohe=False):
    if ohe:
        return np.array([seq.shape[1] for seq in seqs])
    else:
        return np.array([len(seq) for seq in seqs])


def seq_len_sdata(sdata, copy=False):
    if sdata.seqs is not None:
        sdata["seq_len"] = seq_lens(sdata.seqs, ohe=False)
    elif sdata.ohe_seqs is not None:
        sdata["seq_len"] = seq_lens(sdata.ohe_seqs, ohe=True)
    else:
        raise ValueError("No sequences found in sdata")


def gc_content_seq(seq, ohe=False):
    if ohe:
        return np.sum(seq[1:3, :]) / seq.shape[1]
    else:
        return (seq.count("G") + seq.count("C")) / len(seq)


def gc_content_seqs(seqs, ohe=False):
    if ohe:
        seq_len = seqs[0].shape[1]
        return np.sum(seqs[:, 1:3, :], axis=1).sum(axis=1) / seq_len
    else:
        return np.array([gc_content_seq(seq) for seq in seqs])
