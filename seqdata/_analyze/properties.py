def gc_content_seq(seq, ohe=False):
    if ohe:
        return np.sum(seq[1:3, :])/seq.shape[1]
    else:
        return (seq.count("G") + seq.count("C"))/len(seq)
    
    
def gc_content_seqs(seqs, ohe=False):
    if ohe:
        seq_len = seqs[0].shape[1]
        return np.sum(seqs[:, 1:3, :], axis=1).sum(axis=1)/seq_len
    else:
        return np.array([gc_content_seq(seq) for seq in seqs])