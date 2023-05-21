from typing import List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import pysam
import zarr
from more_itertools import split_when
from numcodecs import Blosc, VLenBytes, blosc
from numpy.typing import NDArray
from tqdm import tqdm

from seqdata._io.utils import _get_row_batcher
from seqdata.alphabets import ALPHABETS, SequenceAlphabet
from seqdata.types import FlatReader, PathType, RegionReader

### pysam and cyvcf2 implementation NOTE ###

# pysam.FastaFile.fetch
# contig not found => raises KeyError
# if start < 0 => raises ValueError
# if end > reference length => truncates interval


class FlatFASTA(FlatReader):
    def __init__(
        self,
        name: str,
        fasta: PathType,
        batch_size: int,
        n_threads: int = 1,
        fixed_length: bool = False,
        length_dim: Optional[str] = None,
    ) -> None:
        self.name = name
        self.fasta = fasta
        self.batch_size = batch_size
        self.n_threads = n_threads
        with pysam.FastaFile(str(self.fasta)) as f:
            self.n_seqs = len(f.references)
        self.fixed_length = fixed_length
        self.length_dim = length_dim

    def _reader(self, f: pysam.FastaFile):
        for seq_name in f.references:
            seq = f.fetch(seq_name).encode("ascii")
            yield seq

    def _write_fixed_length(self, out: PathType, overwrite=False):
        blosc.set_nthreads(self.n_threads)
        compressor = Blosc("zstd", clevel=7, shuffle=-1)

        z = zarr.open_group(out)
        with pysam.FastaFile(str(self.fasta)) as f:
            seq_names = f.references
            self.length = f.get_reference_length(seq_names[0])

            arr = z.array(
                "_sequence",
                data=np.array(list(seq_names), object),
                overwrite=overwrite,
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = ["_sequence"]

            n_seqs = len(seq_names)
            batch_size = min(n_seqs, self.batch_size)

            seqs = z.empty(
                self.name,
                shape=(n_seqs, self.length),
                dtype=f"|S{self.length}",
                chunks=batch_size,
                overwrite=overwrite,
                compressor=compressor,
            )
            length_dim = (
                f"{self.name}_length" if self.length_dim is None else self.length_dim
            )
            seqs.attrs["_ARRAY_DIMENSIONS"] = ["_sequence", length_dim]

            batch = np.empty(batch_size, dtype=object)

            row_batcher = _get_row_batcher(self._reader(f), batch_size)
            for last_row, last_in_batch, seq, batch_idx, start_idx in tqdm(
                row_batcher, total=n_seqs
            ):
                if len(seq) != self.length:
                    raise RuntimeError(
                        """
                        Fixed length FlatFASTA reader got sequences with different
                        lengths.
                        """
                    )
                seq = np.frombuffer(seq, f"|S{self.length}")
                batch[batch_idx] = seq
                if last_in_batch or last_row:
                    seqs[start_idx : start_idx + batch_idx + 1] = batch[: batch_idx + 1]

    def _write_variable_length(self, out: PathType, overwrite=False):
        blosc.set_nthreads(self.n_threads)
        compressor = Blosc("zstd", clevel=7, shuffle=-1)

        z = zarr.open_group(out)
        with pysam.FastaFile(str(self.fasta)) as f:
            seq_names = f.references

            arr = z.array(
                f"{self.name}_id",
                data=np.array(list(seq_names), object),
                overwrite=overwrite,
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = ["_sequence"]

            n_seqs = len(seq_names)
            batch_size = min(n_seqs, self.batch_size)

            seqs = z.empty(
                self.name,
                shape=n_seqs,
                dtype=object,
                chunks=batch_size,
                overwrite=overwrite,
                compressor=compressor,
                object_codec=VLenBytes(),
            )
            seqs.attrs["_ARRAY_DIMENSIONS"] = ["_sequence"]

            batch = np.empty(batch_size, dtype=object)

            row_batcher = _get_row_batcher(self._reader(f), batch_size)
            for last_row, last_in_batch, seq, batch_idx, start_idx in tqdm(
                row_batcher, total=n_seqs
            ):
                batch[batch_idx] = seq
                if last_in_batch or last_row:
                    seqs[start_idx : start_idx + batch_idx + 1] = batch[: batch_idx + 1]

    # mock single dispatch on class state
    def _write(self, out: PathType, overwrite=False) -> None:
        if self.fixed_length:
            self._write_fixed_length(out=out, overwrite=overwrite)
        else:
            self._write_variable_length(out=out, overwrite=overwrite)


class GenomeFASTA(RegionReader):
    def __init__(
        self,
        name: str,
        fasta: PathType,
        batch_size: int,
        n_threads: int = 1,
        alphabet: Optional[Union[str, SequenceAlphabet]] = None,
        length_dim: Optional[str] = None,
    ) -> None:
        self.name = name
        self.fasta = fasta
        self.batch_size = batch_size
        self.n_threads = n_threads
        if alphabet is None:
            self.alphabet = ALPHABETS["DNA"]
        elif isinstance(alphabet, str):
            self.alphabet = ALPHABETS[alphabet]
        else:
            self.alphabet = alphabet
        self.length_dim = f"{self.name}_length" if length_dim is None else length_dim

    def _reader(self, bed: pd.DataFrame, f: pysam.FastaFile):
        for row in tqdm(bed.itertuples(index=False), total=len(bed)):
            contig, start, end = cast(Tuple[str, int, int], row[:3])
            seq = f.fetch(contig, start, end).encode("ascii")
            if (pad_len := end - start - len(seq)) > 0:
                pad_left = start < 0
                if pad_left:
                    seq = (b"N" * pad_len) + seq
                else:
                    seq += b"N" * pad_len
            yield seq

    def _spliced_reader(self, bed: pd.DataFrame, f: pysam.FastaFile):
        pbar = tqdm(total=len(bed))
        for rows in split_when(
            bed.itertuples(index=False), lambda x, y: x.name != y.name
        ):
            unspliced: List[bytes] = []
            for row in rows:
                pbar.update()
                contig, start, end = cast(Tuple[str, int, int], row[:3])
                seq = f.fetch(contig, start, end).encode("ascii")
                if (pad_len := end - start - len(seq)) > 0:
                    pad_left = start < 0
                    if pad_left:
                        seq = (b"N" * pad_len) + seq
                    else:
                        seq += b"N" * pad_len
                unspliced.append(seq)
            spliced = b"".join(unspliced)
            yield spliced

    def _write(
        self,
        out: PathType,
        bed: pd.DataFrame,
        length: Optional[int] = None,
        overwrite=False,
        splice=False,
    ) -> None:
        if length is None:
            self._write_variable_length(
                out=out, bed=bed, overwrite=overwrite, splice=splice
            )
        else:
            self._write_fixed_length(
                out=out, bed=bed, length=length, overwrite=overwrite, splice=splice
            )

    def _write_fixed_length(
        self,
        out: PathType,
        bed: pd.DataFrame,
        length: int,
        overwrite: bool,
        splice: bool,
    ):
        blosc.set_nthreads(self.n_threads)
        compressor = Blosc("zstd", clevel=7, shuffle=-1)

        n_seqs = len(bed)
        batch_size = min(n_seqs, self.batch_size)
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        root = zarr.open_group(out)

        seqs = root.empty(
            self.name,
            shape=(n_seqs, length),
            dtype="|S1",
            chunks=(batch_size, None),
            overwrite=overwrite,
            compressor=compressor,
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = ["_sequence", self.length_dim]

        batch = cast(NDArray[np.bytes_], np.empty((batch_size, length), dtype="|S1"))

        with pysam.FastaFile(str(self.fasta)) as f:
            if splice:
                row_batcher = _get_row_batcher(self._spliced_reader(bed, f), batch_size)
            else:
                row_batcher = _get_row_batcher(self._reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, seq, idx, start in tqdm(
                row_batcher, total=n_seqs
            ):
                seq = np.frombuffer(seq, "|S1")
                batch[idx] = seq
                if is_last_in_batch or is_last_row:
                    _batch = batch[: idx + 1]
                    to_rc_mask = to_rc[start : start + idx + 1]
                    _batch[to_rc_mask] = self.alphabet.rev_comp_byte(_batch[to_rc_mask])
                    seqs[start : start + idx + 1] = _batch

    def _write_variable_length(
        self,
        out: PathType,
        bed: pd.DataFrame,
        overwrite: bool,
        splice: bool,
    ):
        blosc.set_nthreads(self.n_threads)
        compressor = Blosc("zstd", clevel=7, shuffle=-1)

        n_seqs = len(bed)
        batch_size = min(n_seqs, self.batch_size)
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        root = zarr.open_group(out)

        seqs = root.empty(
            self.name,
            shape=n_seqs,
            dtype=object,
            chunks=batch_size,
            overwrite=overwrite,
            compressor=compressor,
            object_codec=VLenBytes(),
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = ["_sequence"]

        batch = cast(NDArray[np.object_], np.empty(batch_size, dtype=object))

        with pysam.FastaFile(str(self.fasta)) as f:
            if splice:
                row_batcher = _get_row_batcher(self._spliced_reader(bed, f), batch_size)
            else:
                row_batcher = _get_row_batcher(self._reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, seq, idx, start in tqdm(
                row_batcher, total=n_seqs
            ):
                if to_rc[start + idx]:
                    batch[idx] = self.alphabet.rev_comp_bstring(seq)
                else:
                    batch[idx] = seq
                if is_last_in_batch or is_last_row:
                    seqs[start : start + idx + 1] = batch[: idx + 1]
