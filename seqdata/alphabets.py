from typing import Dict, cast

import numpy as np
from numpy.typing import NDArray


class SequenceAlphabet:
    def __init__(self, alphabet: str, complement: str) -> None:
        """Parse and validate sequence alphabets.

        Nucleic acid alphabets must be complemented by being reversed (without the unknown character).
        For example, `reverse(ACGT) = complement(ACGT) = TGCA`.

        Parameters
        ----------
        alphabet : str
            For example, DNA could be 'ACGT'.
        complement : str
            Complement of the alphabet, to continue the example this would be 'TGCA'.
        """
        self.validate(alphabet, complement)
        self.alphabet = alphabet
        self.complement = complement
        self.array = cast(
            NDArray[np.bytes_], np.frombuffer(self.alphabet.encode("ascii"), "|S1")
        )
        self.complement_map: Dict[str, str] = dict(
            zip(list(self.alphabet), list(self.complement))
        )
        self.complement_map_bytes = {
            k.encode("ascii"): v.encode("ascii") for k, v in self.complement_map.items()
        }
        self.str_comp_table = str.maketrans(self.complement_map)
        self.bytes_comp_table = bytes.maketrans(
            self.alphabet.encode("ascii"), self.complement.encode("ascii")
        )
        self.sorter = np.argsort(self.array)

    def validate(self, alphabet, complement):
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("Alphabet has repeated characters.")

        if len(set(complement)) != len(complement):
            raise ValueError("Complement has repeated characters.")

        for maybe_complement, complement in zip(alphabet[::-1], complement):
            if maybe_complement != complement:
                raise ValueError("Reverse of alphabet does not yield the complement.")

    def bytes_to_ohe(self, arr: NDArray[np.bytes_]) -> NDArray[np.uint8]:
        """Convert an array of byte strings or characters to a one hot encoded array.

        Parameters
        ----------
        arr : ndarray[bytes]
            Array of dtype "|S1"

        Returns
        -------
        ndarray[uint8]
            If arr has shape (a b), this will return an array of shape (a b length_of_alphabet)
        """
        idx = self.sorter[np.searchsorted(self.array[self.sorter], arr)]
        ohe = np.eye(len(self.array), dtype=np.uint8)[idx]
        return ohe

    def ohe_to_bytes(
        self, ohe_arr: NDArray[np.uint8], ohe_axis=-1
    ) -> NDArray[np.bytes_]:
        idx = ohe_arr.nonzero()[ohe_axis]
        if ohe_axis < 0:
            ohe_axis_idx = ohe_arr.ndim + ohe_axis
        else:
            ohe_axis_idx = ohe_axis
        shape = *ohe_arr.shape[:ohe_axis_idx], *ohe_arr.shape[ohe_axis_idx + 1 :]
        return self.array[idx].reshape(shape)

    def complement_bytes(self, byte_arr: NDArray[np.bytes_]) -> NDArray[np.bytes_]:
        """Get reverse complement of byte (string) array.

        Parameters
        ----------
        byte_arr : ndarray[bytes]
            Array of shape `(..., length)` to complement. In other words, elements of
            the array should be single characters.
        """
        # NOTE: a vectorized implementation using np.unique is NOT faster even for
        # longer alphabets like IUPAC DNA/RNA. Another micro-optimization to try would
        # be using vectorized bit manipulations.
        out = byte_arr.copy()
        for nuc, comp in self.complement_map_bytes.items():
            out[byte_arr == nuc] = comp
        return out

    def rev_comp_byte(self, byte_arr: NDArray[np.bytes_]) -> NDArray[np.bytes_]:
        """Get reverse complement of byte (string) array.

        Parameters
        ----------
        byte_arr : ndarray[bytes]
            Array of shape (regions [samples] [ploidy] length) to complement.
        """
        out = self.complement_bytes(byte_arr)
        return out[..., ::-1]

    def rev_comp_string(self, string: str):
        comp = string.translate(self.str_comp_table)
        return comp[::-1]

    def rev_comp_bstring(self, bstring: bytes):
        comp = bstring.translate(self.bytes_comp_table)
        return comp[::-1]


ALPHABETS = {
    "DNA": SequenceAlphabet(alphabet="ACGT", complement="TGCA"),
    "RNA": SequenceAlphabet(alphabet="ACGU", complement="UGCA"),
}
