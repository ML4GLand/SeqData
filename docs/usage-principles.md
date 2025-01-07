# Usage Principles
SeqData is designed to facilitate reading common biological sequence data formats into analysis and machine learning pipelines. The following principles guide its design and usage:

## Unified Interface for Reading Data
SeqData provides a consistent interface for reading two major types of data:

1. **Genomic Sequences:**
   - **TSV:** Explicitly defined sequences in tabular format (e.g., CSV).
   - **FASTA:** Explicitly defined sequences in FASTA format.
   - **BED:** Implicitly defined sequences corresponding to genomic start and end coordinates.

2. **Read Alignment/Coverage Data:**
   - **BAM:** Summarizes read alignments overlapping genomic regions.
   - **BigWig:** Summarizes coverage data overlapping genomic regions.

## Handling Diverse Experimental Data
SeqData accommodates various experimental data types by combining file formats based on the dataset and analysis goals. Common examples in regulatory genomics include:

1. **Massively Parallel Reporter Assays (MPRAs):**
   - Use simple TSV or "flat" FASTA files to store information about regulatory activity. These formats can be read without a reference genome.
   - See [tutorials](tutorials/1_Reading_Flat_Files.md) for details.

2. **ATAC-seq or ChIP-seq Data:**
   - Typically stored in BAM or BigWig files. Combined with a reference genome and coordinates in a BED file, SeqData enables reading DNA sequences and associated read coverage data.
   - See [tutorials](tutorials/2_Reading_Coverage_Data.md).

## Building XArray Datasets Backed by Zarr Stores
SeqData transforms these file formats into Zarr stores that can be read as XArray datasets. XArray provides N-dimensional labeled arrays, similar to NumPy arrays, with the following benefits:

- **Lazy Loading:** Using Dask-backed Zarr stores, SeqData loads and processes only the required subsets of data, making it suitable for large datasets.
- **Efficiency:** Aligns sequences, coverage, and metadata in a unified structure.

See [tutorials](tutorials/3_XArray_Zarr.md) for implementation details.

## Standards Added to XArray Datasets
SeqData enhances XArray datasets with additional standards to better support genomic sequence data:

##tandardized Dimensions:
- `_sequence`: Number of sequences in the dataset.
- `_length`: Length of sequences (exists only for fixed-length sequences).
- `_sample_cov`: Number of coverage tracks (samples)
- `_ohe`: Alphabet size for one-hot encoding.

##ttributes:
- `max_jitter`: Stores maximum jitter information for sequences.

##oordinate Naming Conventions for BED Files:
- `chrom`: Chromosome name.
- `chromStart`: Start coordinate in the reference genome.
- `chromEnd`: End coordinate in the reference genome.

##/O Terminology:
- **Fixed-Length Sequences:**
  - For narrowPeak-like BED files, the fixed length is calculated from the summit (9th column).
  - For other BED files, it is calculated from the midpoint.

## Everything CAN Be Stored in a Single XArray Dataset
SeqData enables storing all relevant data in one XArray dataset, ensuring alignment and accessibility. This unified dataset can include sequences, coverage, metadata, sequence attribution, prediction tracks, and more. With lazy loading, users can selectively access only the data needed for specific analyses, reducing memory overhead.

## Conversion to PyTorch Dataloaders
SeqData simplifies the transition to machine learning workflows by supporting the conversion of XArray datasets to PyTorch dataloaders via the `sd.to_torch_dataloader()` method. This method flexibly handles genomic datasets for sequence-to-function modeling.

See [tutorials](tutorials/4_PyTorch_Dataloaders.md) for usage examples.

