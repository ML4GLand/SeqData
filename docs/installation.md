# Installation
You must have Python version 3.9 or higher installed to use SeqData. SeqData can be installed using `pip`:

```bash
pip install seqdata
```

## Developmental installation
To work with the latest version [on GitHub](https://github.com/ML4GLand/SeqData), clone the repository and `cd` into its root directory.

```bash
git clone https://github.com/ML4GLand/SeqData.git
cd 
```

Then, install the package in development mode:

```bash
pip install -e .
```

## Optional dependencies
If you plan on building PyTorch dataloaders from SeqData objects, you will need to install SeqData with PyTorch:

```bash
pip install seqdata[torch]
```

## Troubleshooting
If you have any issues installing, please [open an issue](https://github.com/ML4GLand/SeqData/issues) on GitHub.
