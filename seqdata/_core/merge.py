from typing import Union, List
import numpy as np
import pandas as pd


def _concat_seqsm(seqsms, keys):
    res = {}
    for i, seqsm in enumerate(seqsms):
        for key in seqsm:
            if key in res:
                if res[key].shape[1] == seqsm[key].shape[1]:
                    res[key] = np.concatenate([res[key], seqsm[key]])
                else:
                    print(f"{keys[i]}'s {key} is not the same shape as previous, skipping")
                    continue
            elif i == 0:
                res[key] = seqsm[key]
            else:
                print(f"{keys[i]} does not contain {key}, skipping {key}")
                continue
    return res

def concat(
    sdatas,
    keys: Union[str, list] = None,
):
    """Concatenates a list of SeqData objects together without merging.

    Does not currently support merging of uns and seqsm.
    Only objects present in the first sdata of the list will be merged

    Parameters
    ----------
    sdatas : list of SeqData objects
        List of SeqData objects to concatenate together
    keys : str or list, optional
        Names to add in seqs_annot column "batch"
    """
    from . import SeqData

    concat_seqs = (
        np.concatenate([s.seqs for s in sdatas]) if sdatas[0].seqs is not None else None
    )
    concat_names = (
        np.concatenate([s.names for s in sdatas])
        if sdatas[0].names is not None
        else None
    )
    concat_ohe_seqs = (
        np.concatenate([s.ohe_seqs for s in sdatas])
        if sdatas[0].ohe_seqs is not None
        else None
    )
    concat_rev_seqs = (
        np.concatenate([s.rev_seqs for s in sdatas])
        if sdatas[0].rev_seqs is not None
        else None
    )
    concat_rev_ohe_seqs = (
        np.concatenate([s.ohe_rev_seqs for s in sdatas])
        if sdatas[0].ohe_rev_seqs is not None
        else None
    )
    concat_seqsm = _concat_seqsm([s.seqsm for s in sdatas], keys=keys)
    for i, s in enumerate(sdatas):
        s["batch"] = keys[i]
    concat_seqs_annot = pd.concat([s.seqs_annot for s in sdatas])
    return SeqData(
        seqs=concat_seqs,
        names=concat_names,
        ohe_seqs=concat_ohe_seqs,
        rev_seqs=concat_rev_seqs,
        ohe_rev_seqs=concat_rev_ohe_seqs,
        seqs_annot=concat_seqs_annot,
        seqsm=concat_seqsm
    )
