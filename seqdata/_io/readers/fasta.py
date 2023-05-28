from typing import List, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import pysam
import seqpro as sp
import zarr
from more_itertools import split_when
from numcodecs import Blosc, VLenBytes, VLenUTF8, blosc
from numpy.typing import NDArray
from tqdm import tqdm

from seqdata._io.utils import _get_row_batcher
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
    ) -> None:
        self.name = name
        self.fasta = fasta
        self.batch_size = batch_size
        self.n_threads = n_threads
        with pysam.FastaFile(str(self.fasta)) as f:
            self.n_seqs = len(f.references)

    def _reader(self, f: pysam.FastaFile):
        for seq_name in f.references:
            seq = f.fetch(seq_name).encode("ascii")
            yield seq

    def _write(
        self,
        out: PathType,
        fixed_length: bool,
        sequence_dim: str,
        length_dim: Optional[str] = None,
        overwrite=False,
    ) -> None:
        if self.name in (sequence_dim, length_dim):
            raise ValueError("Name cannot be equal to sequence_dim or length_dim.")

        blosc.set_nthreads(self.n_threads)
        compressor = Blosc("zstd", clevel=7, shuffle=-1)

        z = zarr.open_group(out)
        with pysam.FastaFile(str(self.fasta)) as f:
            seq_names = f.references
            length = f.get_reference_length(seq_names[0])

            arr = z.array(
                sequence_dim,
                data=np.array(list(seq_names), object),
                overwrite=overwrite,
                object_codec=VLenUTF8(),
            )
            arr.attrs["_ARRAY_DIMENSIONS"] = [sequence_dim]

            n_seqs = len(seq_names)
            batch_size = min(n_seqs, self.batch_size)

            if fixed_length:
                shape = (n_seqs, length)
                dtype = "|S1"
                chunks = (batch_size, None)
                object_codec = None
                seq_dims = [sequence_dim, length_dim]
                batch = np.empty((batch_size, length), dtype="|S1")
            else:
                shape = n_seqs
                dtype = object
                chunks = batch_size
                object_codec = VLenBytes()
                seq_dims = [sequence_dim]
                batch = np.empty(batch_size, dtype=object)

            seqs = z.empty(
                self.name,
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                overwrite=overwrite,
                compressor=compressor,
                object_codec=object_codec,
            )
            seqs.attrs["_ARRAY_DIMENSIONS"] = seq_dims

            row_batcher = _get_row_batcher(self._reader(f), batch_size)
            for last_row, last_in_batch, seq, batch_idx, start_idx in tqdm(
                row_batcher, total=n_seqs
            ):
                if fixed_length and len(seq) != length:
                    raise RuntimeError(
                        """
                        Fixed length FlatFASTA reader got sequences with different
                        lengths.
                        """
                    )
                if fixed_length:
                    seq = np.frombuffer(seq, "|S1")
                batch[batch_idx] = seq
                if last_in_batch or last_row:
                    seqs[start_idx : start_idx + batch_idx + 1] = batch[: batch_idx + 1]


class GenomeFASTA(RegionReader):
    def __init__(
        self,
        name: str,
        fasta: PathType,
        batch_size: int,
        n_threads: int = 1,
        alphabet: Optional[Union[str, sp.NucleotideAlphabet]] = None,
    ) -> None:
        self.name = name
        self.fasta = fasta
        self.batch_size = batch_size
        self.n_threads = n_threads
        if alphabet is None:
            self.alphabet = sp.alphabets.DNA
        elif isinstance(alphabet, str):
            self.alphabet = getattr(sp.alphabets, alphabet)
        else:
            self.alphabet = alphabet

    def _reader(self, bed: pd.DataFrame, f: pysam.FastaFile):
        for row in tqdm(bed.itertuples(index=False), total=len(bed)):
            contig, start, end = cast(Tuple[str, int, int], row[:3])
            seq = f.fetch(contig, max(0, start), end).encode("ascii")
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
                seq = f.fetch(contig, max(0, start), end).encode("ascii")
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
        fixed_length: Union[int, Literal[False]],
        sequence_dim: str,
        length_dim: Optional[str] = None,
        overwrite=False,
        splice=False,
    ) -> None:
        if self.name in (sequence_dim, length_dim):
            raise ValueError("Name cannot be equal to sequence_dim or length_dim.")
        if fixed_length is False:
            self._write_variable_length(
                out=out,
                bed=bed,
                sequence_dim=sequence_dim,
                overwrite=overwrite,
                splice=splice,
            )
        else:
            assert length_dim is not None
            self._write_fixed_length(
                out=out,
                bed=bed,
                fixed_length=fixed_length,
                sequence_dim=sequence_dim,
                length_dim=length_dim,
                overwrite=overwrite,
                splice=splice,
            )

    def _write_fixed_length(
        self,
        out: PathType,
        bed: pd.DataFrame,
        fixed_length: int,
        sequence_dim: str,
        length_dim: str,
        overwrite: bool,
        splice: bool,
    ):
        blosc.set_nthreads(self.n_threads)
        compressor = Blosc("zstd", clevel=7, shuffle=-1)

        if splice:
            n_seqs = bed["name"].nunique()
        else:
            n_seqs = len(bed)
        batch_size = min(n_seqs, self.batch_size)
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        root = zarr.open_group(out)

        seqs = root.empty(
            self.name,
            shape=(n_seqs, fixed_length),
            dtype="|S1",
            chunks=(batch_size, None),
            overwrite=overwrite,
            compressor=compressor,
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = [sequence_dim, length_dim]

        batch = cast(
            NDArray[np.bytes_], np.empty((batch_size, fixed_length), dtype="|S1")
        )

        with pysam.FastaFile(str(self.fasta)) as f:
            if splice:
                row_batcher = _get_row_batcher(self._spliced_reader(bed, f), batch_size)
            else:
                row_batcher = _get_row_batcher(self._reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, seq, idx, start in row_batcher:
                seq = np.frombuffer(seq, "|S1")
                batch[idx] = seq
                if is_last_in_batch or is_last_row:
                    _batch = batch[: idx + 1]
                    to_rc_mask = to_rc[start : start + idx + 1]
                    _batch[to_rc_mask] = self.alphabet.rev_comp_byte(
                        _batch[to_rc_mask], length_axis=-1
                    )
                    seqs[start : start + idx + 1] = _batch

    def _write_variable_length(
        self,
        out: PathType,
        bed: pd.DataFrame,
        sequence_dim: str,
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
        seqs.attrs["_ARRAY_DIMENSIONS"] = [sequence_dim]

        batch = cast(NDArray[np.object_], np.empty(batch_size, dtype=object))

        with pysam.FastaFile(str(self.fasta)) as f:
            if splice:
                row_batcher = _get_row_batcher(self._spliced_reader(bed, f), batch_size)
            else:
                row_batcher = _get_row_batcher(self._reader(bed, f), batch_size)
            for is_last_row, is_last_in_batch, seq, idx, start in row_batcher:
                if to_rc[start + idx]:
                    batch[idx] = self.alphabet.rev_comp_bstring(seq)
                else:
                    batch[idx] = seq
                if is_last_in_batch or is_last_row:
                    seqs[start : start + idx + 1] = batch[: idx + 1]
