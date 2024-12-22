# Usage Principles
This page outlines the 3 key principles of SeqData's design that dictate how it should be used.

## 1. SeqData primarily works with 2 kinds of data:

1. Tabular CSV/flat FASTA files with full sequences

2. Region BED files from a reference FASTA

## 2. SeqData builds XArray Datasets backed by Zarr stores

## 3. SeqData can convert XArray Datasets to PyTorch Dataloaders generalizably

## 4. Everything CAN be stored in a single XArrray dataset

## 5. SeqData defaults

Dimensions:
 - "_sequence" -- sequence dimension corresponding to the number of sequences in the dataset
 - "_length" -- length dimension corresponding to the length of the sequences in the dataset. Will only exist for fixed length sequences.
 - "_cov" -- coverage dimension corresponding to the number of coverages in the dataset
 - "_ohe" -- one-hot encoding dimension corresponding to the alphabet size of the sequences in the dataset

Regions:
 - "chrom
 - "chromStart"
 - "chromEnd"

IO:
 - If BED file is narrowPeak like, the fixed length will be calculated from the summit (9th column). Otherwise it will be calculated from the midpoint


# SeqData provides a unified interface for loading two major types of data:

1. Genomic sequences:
* **TSV** -- explicitly defined sequences in tabular format (e.g. CSV)
* **FASTA** -- explicitly defined sequences in FASTA format
* **BED** -- implicitly define sequences corresponding to genomic start and end coordinates

2. Read alignment/coverage data (paired with a reference genome):
* **BAM** -- summarizes read alignments that overlap a genomic region
* **BigWig** -- summarizes coverage data that overlaps a genomic region