```{eval-rst}
.. module:: seqdata
```

```{eval-rst}
.. automodule:: seqdata
   :noindex:
```

# API

## Readers
These classes are designed to read data from a variety of file formats into a SeqData object.

```{eval-rst}
.. autosummary::
   :toctree: api/

   BAM
   VCF
   BigWig
   FlatFASTA
   GenomeFASTA
   Table
```

## Composing readers
These functions are designed to be used in a composable way to read data from a variety of file formats into a single SeqData object.

```{eval-rst}
.. autosummary::
   :toctree: api/

   from_flat_files
   from_region_files
```

## Default readers
These functions are special cases of the composable readers that are designed to be used for common use cases

```{eval-rst}
.. autosummary::
   :toctree: api/

   read_bam
   read_bigwig
   read_flat_fasta
   read_genome_fasta
   read_table
   read_vcf
   read_bedlike
```

## Zarr
SeqData reads and writes all datasets to disk as Zarr stores using the following functions

```{eval-rst}
.. autosummary::
   :toctree: api/

   to_zarr
   open_zarr
``` 

## PyTorch dataloading
SeqData provides a unified interface for converting SeqData objects into PyTorch dataloaders

```{eval-rst}
.. autosummary::
   :toctree: api/

   get_torch_dataloader
```

## Utilities
Some utility functions that are useful for working with SeqData objects

```{eval-rst}
.. autosummary::
   :toctree: api/

   add_bed_to_sdata
   label_overlapping_regions
   merge_obs
```
