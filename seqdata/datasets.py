from __future__ import annotations
from typing_extensions import Literal

from pathlib import Path
import pooch
import tarfile


# This is a global variable used to store all datasets. It is initialized only once
# when the data is requested.
_datasets = None

def datasets():
    global _datasets
    if _datasets is None:
        _datasets = pooch.create(
            path=pooch.os_cache("seqdata"),
            base_url="https://zenodo.org/records/11415225/files/",
            env="SEQDATA_DATA_DIR",  # The user can overwrite the storage path by setting this environment variable.
            registry={
                # K562 ATAC-seq
                "K562_ATAC-seq.zarr.tar.gz": "sha256:da601746f933a623fc0465c172f0338425690d480ae4aa7c6d645f02d32a7504",
                "signal.bw": "sha256:df4b2af6ad7612207dcb4f6acce41e8f731b08d2d84c00263f280325c9be8f53",

                # K562 CTCF ChIP-seq
                "K562_CTCF-ChIP-seq.zarr.tar.gz": "sha256:c0098fce7464459e88c8b1ef30cad84c1931b818273c07f80005eeb9037e8276",
                "plus.bw": "sha256:005ba907136c477754c287113b3479a68121c47368455fef9f19f593e2623462",
                "minus.bw": "sha256:2ff74b44bea80b1c854a265a1f759a3e1aa7baec10ba20139e39d78d7ea5e1ed",

                # TF motifs
                "cisBP_human.meme": "sha256:<new_sha256_hash_for_cisBP_human.meme>",
                "Meuleman_2020.meme": "sha256:<new_sha256_hash_for_Meuleman_2020.meme>",

                # Genome files
                "gencode_v41_GRCh38.fa.gz": "sha256:<new_sha256_hash_for_gencode_v41_GRCh38.fa.gz>",
            },
            urls={
                "K562_ATAC-seq.zarr": "https://zenodo.org/records/11415225/files/K562_ATAC-seq.zarr",
                "signal.bw": "https://zenodo.org/records/11415225/files/signal.bw",
                "K562_CTCF-ChIP-seq.zarr": "https://zenodo.org/records/11415225/files/K562_CTCF-ChIP-seq.zarr",
                "plus.bw": "https://zenodo.org/records/11415225/files/plus.bw",
                "minus.bw": "https://zenodo.org/records/11415225/files/minus.bw",
                "cisBP_human.meme": "https://zenodo.org/records/11415225/files/cisBP_human.meme",
                "Meuleman_2020.meme": "https://zenodo.org/records/11415225/files/Meuleman_2020.meme",
                "gencode_v41_GRCh38.fa.gz": "https://zenodo.org/records/11415225/files/gencode_v41_GRCh38.fa.gz",
            },
        )
    return _datasets


def K562_ATAC_seq(type: Literal["seqdata", "bigwig"]="seqdata") -> Path:
    if type == "seqdata":
        path = Path(datasets().fetch("K562_ATAC-seq.zarr.tar.gz"))
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(path.parent)
        path.unlink()  # Remove the tar.gz file after extraction
        extracted_path = path.parent / "K562_ATAC-seq.zarr"
        return extracted_path
    elif type == "bigwig":
        return Path(datasets().fetch("signal.bw"))
    
def K562_CTCF_ChIP_seq(type: Literal["seqdata", "bigwig"]="seqdata") -> Path:
    if type == "seqdata":
        path = Path(datasets().fetch("K562_CTCF-ChIP-seq.zarr.tar.gz"))
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(path.parent)
        path.unlink()  # Remove the tar.gz file after extraction
        extracted_path = path.parent / "K562_CTCF-ChIP-seq.zarr"
        return extracted_path
    elif type == "bigwig":
        return Path(datasets().fetch("plus.bw")), Path(datasets().fetch("minus.bw"))
    