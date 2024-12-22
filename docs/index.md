```{toctree}
:hidden: true
:caption: Contents
:maxdepth: 2

installation
usage-principles
tutorials/1_Reading_Tables
api
release-notes
contributing
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
