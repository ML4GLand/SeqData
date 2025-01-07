```{toctree}
:hidden: true
:caption: Contents
:maxdepth: 2

installation
usage-principles
api
release-notes
contributing
references
```

```{toctree}
:hidden: true
:caption: Tutorials
:maxdepth: 2

tutorials/1_Reading_Flat_Files
tutorials/2_Reading_Region_Files
tutorials/3_Reading_Tracks
tutorials/4_Zarr_And_XArray
tutorials/5_PyTorch_Dataloading
tutorials/6_Complex_Transforms
```


# SeqData -- Annotated biological sequence data
```{image} https://badge.fury.io/py/SeqData.svg
:alt: PyPI version
:target: https://badge.fury.io/py/SeqData
:class: inline-link
```

```{image} https://readthedocs.org/projects/SeqData/badge/?version=latest
:alt: Documentation Status
:target: https://SeqData.readthedocs.io/en/latest/index.html
:class: inline-link
```

```{image} https://img.shields.io/pypi/dm/SeqData
:alt: PyPI - Downloads
:class: inline-link
```

SeqData is a Python package for preparing ML-ready genomic sequence datasets. Some of the key features of SeqData include:

- Keeps multi-dimensional data in one object (e.g. sequence, coverage, metadata, etc.)
- Efficiently and flexibly loads of track-based data from BigWig or BAM
- Fully compatible with PyTorch dataloading
- Offers out-of-core dataloading from disk to CPU to GPU

SeqData is designed to be used via its Python API.

# Getting started
* {doc}`Install SeqData <installation>`
* Browse the main {doc}`API <api>`

# Contributing
SeqData is an open-source project and we welcome {doc}`contributions <contributing>`.
