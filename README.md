[![PyPI version](https://badge.fury.io/py/seqexplainer.svg)](https://badge.fury.io/py/seqdata)
![PyPI - Downloads](https://img.shields.io/pypi/dm/seqdata)

<img src="docs/_static/seqdata_xr.png" alt="seqdata xr" width=350>

# SeqData (Annotated sequence data)

[[documentation](https://seqdata.readthedocs.io/en/latest/)][[tutorials]()]

SeqData is a Python package for preparing ML-ready genomic sequence datasets. Some of the key features of SeqData include:

- Keeps multi-dimensional data in one object (e.g. sequence, coverage, metadata, etc.)
- Efficiently and flexibly loads of track-based data from BigWig or BAM
- Fully compatible with PyTorch dataloading
- Offers out-of-core dataloading from disk to CPU to GPU

> [!NOTE] 
> SeqData is under active development. The API has largely been decided on, but may change slightly across versions until the first major release.

## Installation

`pip install seqdata`

## Roadmap

Although my focus will largely follow my research projects and the feedback I receive from the community, here is a roadmap for what I currently plan to focus on in the next few releases.

- v0.1.0: ✔️ Initial API for reading BAM, FASTA, BigWig and Tabular data and building loading PyTorch dataloaders
- v0.2.0: (WIP) Bug fixes, improved documentation, tutorials, and examples
- v0.3.0: Improved out of core functionality, robust BED classification datasets
- v0.0.4 — Interoperability with AnnData and SnapATAC2

## Usage

### Loading data from "flat" files
The simplest way to store genomic sequence data is in a table or in a "flat" fasta file. Though this can easily be accomplished using something like `pandas.read_csv`, the SeqData interface keeps the resulting on-disk and in-memory objects standardized with the rest of the SeqData and larger ML4GLand API.

```python
from seqdata import read_table
sdata = sd.read_table(
    name="seq",  # name of resulting xarray variable containing sequences
    out="sdata.zarr",  # output file
    tables=["sequences.tsv"],  # list of tabular files
    seq_col="seq_col",  # column containing sequences
    fixed_length=False,  # whether all sequences are the same length
    batch_size=1000,  # number of sequences to load at once
    overwrite=True,  # overwrite the output file if it exists
)
```

Will generate a `sdata.zarr` file containing the sequences in the `seq_col` column of `sequences.tsv`. The resulting `sdata` object can then be used for downstream analysis.

### Loading sequences from genomic coordinates

### Loading data from BAM files
Reading from bam files allows one to choose custom counting strategies (often necessary with ATAC-seq data). 

```python
from seqdata import read_bam
sdata = sd.read_bam(
    name="seq",  # name of resulting xarray variable containing sequences
    out="sdata.zarr",  # output file
    bams=["data.bam"],  # list of BAM files
    seq_col="seq_col",  # column containing sequences
    fixed_length=False,  # whether all sequences are the same length
    batch_size=1000,  # number of sequences to load at once
    overwrite=True,  # overwrite the output file if it exists
)
```

### Loading data from BigWig files
[BigWig files](https://genome.ucsc.edu/goldenpath/help/bigWig.html) are a common way to store track-based data and the workhorse of modern genomic sequence based ML. ...

```python
from seqdata import read_bigwig
sdata = sd.read_bigwig(
    name="seq",  # name of resulting xarray variable containing sequences
    out="sdata.zarr",  # output file
    bigwigs=["data.bw"],  # list of BigWig files
    seq_col="seq_col",  # column containing sequences
    fixed_length=False,  # whether all sequences are the same length
    batch_size=1000,  # number of sequences to load at once
    overwrite=True,  # overwrite the output file if it exists
)
```

### Working with Zarr stores and XArray objects
The SeqData API is built to convert data from common formats to Zarr stores on disk. The Zarr store... When coupled with XArray and Dask, we also have the ability to lazy load data and work with data that is too large to fit in memory. 

```python
```

Admittedly, working with XArray can take some getting used to...

### Building a dataloader
The main goal of SeqData is to allow a seamless flow

## Contributing
This section was modified from https://github.com/pachterlab/kallisto.

All contributions, including bug reports, documentation improvements, and enhancement suggestions are welcome. Everyone within the community is expected to abide by our [code of conduct](https://github.com/ML4GLand/EUGENe/blob/main/CODE_OF_CONDUCT.md)

As we work towards a stable v1.0.0 release, and we typically develop on branches. These are merged into `dev` once sufficiently tested. `dev` is the latest, stable, development branch. 

`main` is used only for official releases and is considered to be stable. If you submit a pull request, please make sure to request to merge into `dev` and NOT `main`.
