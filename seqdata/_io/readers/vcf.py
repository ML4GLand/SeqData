import warnings
from pathlib import Path
from typing import List, Literal, Optional, Set, Union, cast

import cyvcf2
import numpy as np
import pandas as pd
import pysam
import seqpro as sp
import zarr
from more_itertools import split_when
from natsort import natsorted
from numcodecs import Blosc, VLenBytes, VLenUTF8, blosc
from numpy.typing import NDArray
from tqdm import tqdm

from seqdata._io.utils import _get_row_batcher
from seqdata.types import PathType, RegionReader

N_HAPLOTYPES = 2


### pysam and cyvcf2 implementation NOTE ###

# pysam.FastaFile.fetch
# contig not found => raises KeyError
# if start < 0 => raises ValueError
# if end > reference length => truncates interval

# cyvcf2.VCF
# Contig not found => warning
# start < 0 => warning
# start = 0 (despite being 1-indexed) => nothing
# end > contig length => treats end = contig length


class VCF(RegionReader):
    name: str
    vcf: Path
    fasta: Path
    contigs: List[str]

    def __init__(
        self,
        name: str,
        vcf: PathType,
        fasta: PathType,
        batch_size: int,
        samples: Optional[List[str]] = None,
        n_threads=1,
        samples_per_chunk=10,
        alphabet: Optional[Union[str, sp.NucleotideAlphabet]] = None,
        sample_dim: Optional[str] = None,
        haplotype_dim: Optional[str] = None,
    ) -> None:
        self.name = name
        self.vcf = Path(vcf)
        self.fasta = Path(fasta)
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.samples_per_chunk = samples_per_chunk
        if alphabet is None:
            self.alphabet = sp.alphabets.DNA
        elif isinstance(alphabet, str):
            self.alphabet = getattr(sp.alphabets, alphabet)
        else:
            self.alphabet = alphabet
        self.sample_dim = f"{name}_sample" if sample_dim is None else sample_dim
        self.haplotype_dim = (
            f"{name}_haplotype" if haplotype_dim is None else haplotype_dim
        )

        with pysam.FastaFile(str(fasta)) as f:
            fasta_contigs = set(f.references)
        _vcf = cyvcf2.VCF(str(vcf), samples=samples)
        self.samples = _vcf.samples if samples is None else samples
        try:
            vcf_contigs = cast(Set[str], set(_vcf.seqlens))
        except AttributeError:
            warnings.warn("VCF header has no contig annotations.")
            vcf_contigs: Set[str] = set()
        _vcf.close()

        self.contigs = cast(List[str], natsorted(fasta_contigs | vcf_contigs))
        if len(self.contigs) == 0:
            raise RuntimeError("FASTA has no contigs.")
        # * don't check for contigs exclusive to FASTA because VCF is not guaranteed to
        # * have any contigs listed
        contigs_exclusive_to_vcf = natsorted(vcf_contigs - fasta_contigs)
        if contigs_exclusive_to_vcf:
            warnings.warn(
                f"VCF has contigs not found in FASTA: {contigs_exclusive_to_vcf}"
            )

    def _get_pos_bases(self, v):
        # change to bytes and extract alleles
        alleles = v.gt_bases.astype("S").reshape(-1, 1).view("S1")[:, [0, 2]]
        # change unknown to reference
        alleles[alleles == "."] = v.REF
        # make position 0-indexed
        return v.POS - 1, alleles

    def _reader(
        self,
        bed: pd.DataFrame,
        f: pysam.FastaFile,
        vcf: cyvcf2.VCF,
        sample_order: NDArray[np.intp],
    ):
        for row in tqdm(bed.itertuples(index=False), total=len(bed)):
            contig, start, end = row[:3]
            start, end = cast(int, start), cast(int, end)
            seq_bytes = f.fetch(contig, max(start, 0), end).encode("ascii")
            pad_left = -min(start, 0)
            pad_right = end - start - len(seq_bytes) - pad_left
            seq_bytes = b"N" * pad_left + seq_bytes + b"N" * pad_right
            seq = cast(NDArray[np.bytes_], np.array([seq_bytes], "S").view("S1"))
            # (samples haplotypes length)
            tiled_seq = np.tile(seq, (len(self.samples), 2, 1))

            region = f"{contig}:{max(start, 0)+1}-{end}"
            positions_ls, alleles_ls = zip(
                *[self._get_pos_bases(v) for v in vcf(region) if v.is_snp]
            )
            # (variants)
            relative_positions = cast(NDArray[np.int64], np.array(positions_ls)) - start
            # (samples haplotypes variants)
            alleles = cast(NDArray[np.bytes_], np.stack(alleles_ls, -1)[sample_order])
            # (samples haplotypes variants) = (samples haplotypes variants)
            tiled_seq[..., relative_positions] = alleles
            # (samples haplotypes length)
            yield tiled_seq

    def _spliced_reader(
        self,
        bed: pd.DataFrame,
        f: pysam.FastaFile,
        vcf: cyvcf2.VCF,
        sample_order: NDArray[np.intp],
    ):
        pbar = tqdm(total=len(bed))
        for rows in split_when(
            bed.itertuples(index=False), lambda x, y: x.name != y.name
        ):
            unspliced: List[NDArray[np.bytes_]] = []
            for row in rows:
                pbar.update()
                contig, start, end = row[:3]
                start, end = cast(int, start), cast(int, end)
                seq_bytes = f.fetch(contig, max(start, 0), end).encode("ascii")
                pad_left = -min(start, 0)
                pad_right = end - start - len(seq_bytes) - pad_left
                seq_bytes = b"N" * pad_left + seq_bytes + b"N" * pad_right
                seq = cast(NDArray[np.bytes_], np.frombuffer(seq_bytes, "|S1"))
                # (samples haplotypes length)
                tiled_seq = np.tile(seq, (len(self.samples), 2, 1))

                region = f"{contig}:{max(start, 0)+1}-{end}"
                positions_ls, alleles_ls = zip(
                    *[self._get_pos_bases(v) for v in vcf(region) if v.is_snp]
                )
                # (variants)
                relative_positions = (
                    cast(NDArray[np.int64], np.array(positions_ls)) - start
                )
                # (samples haplotypes variants)
                alleles = cast(
                    NDArray[np.bytes_], np.stack(alleles_ls, -1)[sample_order]
                )
                # (samples haplotypes variants) = (samples haplotypes variants)
                tiled_seq[..., relative_positions] = alleles
                unspliced.append(tiled_seq)
            # list of (samples haplotypes length)
            yield np.concatenate(unspliced, -1)

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
        if self.name in (sequence_dim, self.sample_dim, self.haplotype_dim, length_dim):
            raise ValueError(
                """Name cannot be equal to sequence_dim, sample_dim, haplotype_dim, or 
                length_dim."""
            )

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

        n_seqs = bed["name"].nunique() if splice else len(bed)
        batch_size = min(n_seqs, self.batch_size)

        z = zarr.open_group(out)

        seqs = z.empty(
            self.name,
            shape=(n_seqs, len(self.samples), N_HAPLOTYPES, fixed_length),
            dtype="|S1",
            chunks=(batch_size, self.samples_per_chunk, 1, None),
            overwrite=overwrite,
            compressor=compressor,
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = [
            sequence_dim,
            self.sample_dim,
            self.haplotype_dim,
            length_dim,
        ]

        arr = z.array(
            self.sample_dim,
            np.array(self.samples, object),
            compressor=compressor,
            overwrite=overwrite,
            object_codec=VLenUTF8(),
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = [self.sample_dim]

        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        _vcf = cyvcf2.VCF(
            self.vcf, lazy=True, samples=self.samples, threads=self.n_threads
        )
        *_, sample_order = np.intersect1d(
            _vcf.samples, self.samples, assume_unique=True, return_indices=True
        )

        # (batch samples haplotypes length)
        batch = cast(
            NDArray[np.bytes_], np.empty((batch_size, *seqs.shape[1:]), dtype="|S1")
        )

        with pysam.FastaFile(str(self.fasta)) as f:
            if splice:
                reader = self._spliced_reader
            else:
                reader = self._reader
            row_batcher = _get_row_batcher(
                reader(bed, f, _vcf, sample_order), batch_size
            )
            for is_last_row, is_last_in_batch, seq, idx, start in row_batcher:
                # (samples haplotypes length)
                batch[idx] = seq
                if is_last_in_batch or is_last_row:
                    _batch = batch[: idx + 1]
                    to_rc_mask = to_rc[start : start + idx + 1]
                    _batch[to_rc_mask] = self.alphabet.rev_comp_byte(
                        _batch[to_rc_mask], length_axis=-1
                    )
                    seqs[start : start + idx + 1] = _batch[: idx + 1]

        _vcf.close()

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

        n_seqs = bed["name"].nunique() if splice else len(bed)
        batch_size = min(n_seqs, self.batch_size)

        z = zarr.open_group(out)

        seqs = z.empty(
            self.name,
            shape=(n_seqs, len(self.samples), N_HAPLOTYPES),
            dtype=object,
            chunks=(batch_size, self.samples_per_chunk, 1),
            overwrite=overwrite,
            compressor=compressor,
            object_codec=VLenBytes(),
        )
        seqs.attrs["_ARRAY_DIMENSIONS"] = [
            sequence_dim,
            self.sample_dim,
            self.haplotype_dim,
        ]

        arr = z.array(
            self.sample_dim,
            np.array(self.samples, object),
            compressor=compressor,
            overwrite=overwrite,
            object_codec=VLenUTF8(),
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = [self.sample_dim]

        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        _vcf = cyvcf2.VCF(
            self.vcf, lazy=True, samples=self.samples, threads=self.n_threads
        )
        *_, sample_order = np.intersect1d(
            _vcf.samples, self.samples, assume_unique=True, return_indices=True
        )

        # (batch samples haplotypes)
        batch = cast(
            NDArray[np.object_], np.empty((batch_size, *seqs.shape[1:]), dtype=object)
        )

        with pysam.FastaFile(str(self.fasta)) as f:
            if splice:
                reader = self._spliced_reader
            else:
                reader = self._reader
            row_batcher = _get_row_batcher(
                reader(bed, f, _vcf, sample_order), batch_size
            )
            for is_last_row, is_last_in_batch, seq, idx, start in row_batcher:
                # (samples haplotypes length)
                if to_rc[idx]:
                    seq = self.alphabet.rev_comp_byte(seq, length_axis=-1)
                # (samples haplotypes)
                batch[idx] = seq.view(f"|S{seq.shape[-1]}").squeeze().astype(object)
                if is_last_in_batch or is_last_row:
                    seqs[start : start + idx + 1] = batch[: idx + 1]

        _vcf.close()

    def _sequence_generator(self, bed: pd.DataFrame, splice=False):
        to_rc = cast(NDArray[np.bool_], (bed["strand"] == "-").to_numpy())

        _vcf = cyvcf2.VCF(
            self.vcf, lazy=True, samples=self.samples, threads=self.n_threads
        )
        *_, sample_order = np.intersect1d(
            _vcf.samples, self.samples, assume_unique=True, return_indices=True
        )

        with pysam.FastaFile(str(self.fasta)) as f:
            if splice:
                reader = self._spliced_reader
            else:
                reader = self._reader
            # (samples haplotypes length)
            for i, seqs in enumerate(reader(bed, f, _vcf, sample_order)):
                if to_rc[i]:
                    seqs = self.alphabet.rev_comp_byte(seqs, length_axis=-1)
                seqs = seqs.view(f"|S{seqs.shape[-1]}")
                for seq in seqs.ravel():
                    yield seq
