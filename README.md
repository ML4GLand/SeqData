[![PyPI version](https://badge.fury.io/py/seqexplainer.svg)](https://badge.fury.io/py/seqdata)
![PyPI - Downloads](https://img.shields.io/pypi/dm/seqdata)

<img src="docs/_static/SeqData_only_v2.png" alt="SeqData Logo" width=350>

# SeqData -- Annotated DNA/RNA sequence data

```python
import seqdata as sd
```

# Core functionality

## I/0
SeqData is designed to handle two major types of data

1. Actual string sequences with annotations
Loaded from FASTA or TSV files
```python
sdata = sd.read_fasta()
sdata = sd.read_tsv()
```

2. Genomic intervals with annotations
Loaded from bed files
```python
sdata = sd.read_bed()
sdata
```

3. SeqData on disk representation
h5
```python
sdata.write_h5sd()
sd.read_h5sd()
```

## Sequence analysis

### Calculate sequence properties (e.g. GC content

### Visaulize sequence properties

### Motif analysis with [MotifData](https://github.com/ML4GLand/MotifData)

  - Perform HOMER motif analysis
  - Perform MEME motif analysis
  - Perform DEM (Cluster-buster based) motif analysis
  - Perform cisTarget
        
## Sequence preprocessing

### Encoding sequences
  
  - one-hot encodings (multiple orders)
  - k-mer frequencies (full and gapped k-mers)
  
## Sequence annotations

### Overlap with known genomic features
    - Known genomic features
    - Overlap with different epigenomics data
