[![PyPI version](https://badge.fury.io/py/seqexplainer.svg)](https://badge.fury.io/py/seqdata)
![PyPI - Downloads](https://img.shields.io/pypi/dm/seqdata)

<img src="docs/_static/SeqData_only_v2.png" alt="SeqData Logo" width=350>

# SeqData -- Annotated biological sequence data

## Installation

1. Clone this repo
2. Use the `environment.yml` file to install explicit dependencies
3. Install [SeqPro](https://github.com/ML4GLand/SeqPro)
4. Install PyTorch as an optional dependency -- enables use of `seqdata.get_torch_dataloader` to get DataLoaders from `xarray.Datasets` (i.e. SeqData objects).