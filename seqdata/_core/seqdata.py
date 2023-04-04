from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numcodecs import Blosc
from numpy.typing import NDArray
from typing_extensions import Self

from seqdata._io.utils import (
    _df_to_xr_zarr,
    _read_bedlike,
    _set_uniform_length_around_center,
)
from seqdata.types import FlatReader, PathType, RegionReader

from .utils import (
    _filter_by_exact_dims,
    _filter_layers,
    _filter_obsm,
    _filter_uns,
    _filter_varm,
    cartesian_product,
)

try:
    import torch
    from torch.utils.data import DataLoader, Sampler

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SeqData:
    ds: xr.Dataset
    path: Optional[Path]
    max_jitter: Optional[int]

    def __init__(self, ds: xr.Dataset) -> None:
        self.ds = ds.copy()

    def sel(
        self,
        indexers: Optional[Mapping[Any, Any]] = None,
        method: Optional[
            Literal["nearest", "pad", "ffill", "backfill", "bfill"]
        ] = None,
        tolerance: Optional[Union[int, float, Iterable[Union[int, float]]]] = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> "SeqData":
        return SeqData(
            self.ds.sel(
                indexers=indexers,
                method=method,
                tolerance=tolerance,
                drop=drop,
                **indexers_kwargs,
            )
        )

    def isel(
        self,
        indexers: Optional[Mapping[Any, Any]] = None,
        drop: bool = False,
        missing_dims: Literal["raise", "warn", "ignore"] = "raise",
        **indexers_kwargs: Any,
    ) -> "SeqData":
        return SeqData(self.ds.isel(indexers, drop, missing_dims, **indexers_kwargs))

    @property
    def layers(self):
        return _filter_layers(self.ds)

    @property
    def obs(self):
        return _filter_by_exact_dims(self.ds, "sequence")

    @property
    def var(self):
        return _filter_by_exact_dims(self.ds, "length")

    @property
    def obsm(self):
        return _filter_obsm(self.ds)

    @property
    def varm(self):
        return _filter_varm(self.ds)

    @property
    def obsp(self):
        return _filter_by_exact_dims(self.ds, ("sequence", "sequence"))

    @property
    def varp(self):
        return _filter_by_exact_dims(self.ds, ("length", "length"))

    @property
    def uns(self):
        return _filter_uns(self.ds)

    def __repr__(self) -> str:
        msg = "SeqData holding an Xarray dataset.\n"
        msg += repr(self.ds)
        return msg

    def _repr_html_(self) -> str:
        msg = "SeqData holding an Xarray dataset.\n"
        msg += self.ds._repr_html_()
        return msg

    def load(self, **kwargs):
        """Load this SeqData into memory, if it isn't already."""
        self.ds = self.ds.load(**kwargs)

    def to_zarr(
        self,
        store: Optional[Union[MutableMapping, PathType]] = None,
        chunk_store: Optional[Union[MutableMapping, PathType]] = None,
        mode: Optional[Literal["w", "w-", "a", "r+"]] = None,
        synchronizer=None,
        group: Optional[str] = None,
        encoding: Optional[Mapping] = None,
        compute=True,
        append_dim: Optional[Hashable] = None,
        region: Optional[Mapping[str, slice]] = None,
        safe_chunks=True,
        storage_options: Optional[Dict[str, str]] = None,
        zarr_version: Optional[int] = None,
    ):
        self.ds.to_zarr(
            store=store,
            chunk_store=chunk_store,
            mode=mode,
            synchronizer=synchronizer,
            group=group,
            encoding=encoding,
            compute=compute,  # type: ignore
            consolidated=True,
            append_dim=append_dim,
            region=region,
            safe_chunks=safe_chunks,
            storage_options=storage_options,
            zarr_version=zarr_version,
        )

    @classmethod
    def open_zarr(
        cls,
        store: PathType,
        group: Optional[str] = None,
        synchronizer=None,
        chunks: Optional[
            Union[Literal["auto"], int, Mapping[str, int], Tuple[int, ...]]
        ] = "auto",
        decode_cf=True,
        mask_and_scale=True,
        decode_times=True,
        decode_coords=True,
        drop_variables: Optional[Union[str, Iterable[str]]] = None,
        consolidated: Optional[bool] = None,
        overwrite_encoded_chunks=False,
        chunk_store: Optional[Union[MutableMapping, PathType]] = None,
        storage_options: Optional[Dict[str, str]] = None,
        decode_timedelta: Optional[bool] = None,
        use_cftime: Optional[bool] = None,
        zarr_version: Optional[int] = None,
        **kwargs,
    ):
        ds = xr.open_zarr(
            store=store,
            group=group,
            synchronizer=synchronizer,
            chunks=chunks,  # type: ignore
            decode_cf=decode_cf,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=False,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            consolidated=consolidated,
            overwrite_encoded_chunks=overwrite_encoded_chunks,
            chunk_store=chunk_store,
            storage_options=storage_options,
            decode_timedelta=decode_timedelta,
            use_cftime=use_cftime,
            zarr_version=zarr_version,
            **kwargs,
        )
        self = cls(ds)
        self.path = Path(store)
        self.max_jitter = cast(int, self.ds.attrs["max_jitter"])
        return self

    @classmethod
    def from_files(
        cls,
        *readers: Union[FlatReader, RegionReader],
        path: PathType,
        length: Optional[int] = None,
        bed: Optional[Union[PathType, pd.DataFrame]] = None,
        max_jitter: int = 0,
        overwrite=False,
    ) -> Self:
        z = zarr.open_group(path)
        z.attrs["max_jitter"] = max_jitter

        if bed is not None and length is not None:
            length += 2 * max_jitter
            if isinstance(bed, (str, Path)):
                _bed = _read_bedlike(bed)
            else:
                _bed = bed
            _set_uniform_length_around_center(_bed, length)
            _df_to_xr_zarr(
                _bed,
                path,
                ["sequence"],
                compressor=Blosc("zstd", clevel=7, shuffle=-1),
                overwrite=overwrite,
            )
        elif bed is not None or length is not None:
            raise ValueError("bed and length are mutually inclusive arguments.")
        else:
            if any(map(lambda r: isinstance(r, RegionReader), readers)):
                raise ValueError(
                    "Got readers that need bed and length but didn't get bed and length."
                )

        for reader in readers:
            if isinstance(reader, FlatReader):
                reader._write(path, overwrite)
            elif isinstance(reader, RegionReader):
                reader._write(
                    path,
                    length,  # type: ignore
                    _bed,  # type: ignore
                    overwrite,
                )

        zarr.consolidate_metadata(path)  # type: ignore

        self = cls.open_zarr(path)
        return self

    def add_layers_from_files(
        self,
        *readers: Union[FlatReader, RegionReader],
        overwrite=False,
    ):
        if self.path is None:
            raise ValueError("SeqData must be backed on disk to add layers from files.")

        if any(map(lambda r: isinstance(r, RegionReader), readers)):
            length = self.ds.sizes["length"]
            bed = self.ds[["chrom", "chromStart", "chromEnd", "strand"]].to_dataframe()

        for reader in readers:
            if isinstance(reader, FlatReader):
                if (
                    reader.n_seqs is not None
                    and reader.n_seqs != self.ds.sizes["sequence"]
                ):
                    raise ValueError(
                        f'Reader "{reader.name}" has a different number of sequences than this SeqData.'
                    )
                reader._write(self.path, overwrite)
            elif isinstance(reader, RegionReader):
                reader._write(
                    self.path,
                    length,  # type: ignore
                    bed,  # type: ignore
                    overwrite,
                )

        self = SeqData.open_zarr(self.path)
        return self

    def get_torch_dataloader(
        self,
        sample_dims: Union[str, List[str]],
        variables: List[str],
        transforms: Optional[Dict[str, Callable[[NDArray], "torch.Tensor"]]] = None,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Union["Sampler", Iterable]] = None,
        batch_sampler: Optional[Union["Sampler[Sequence]", Iterable[Sequence]]] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ) -> "DataLoader":
        """Get a PyTorch DataLoader for this SeqData.

        Parameters
        ----------
        sample_dims : str or list[str]
            Sample dimensions will be sliced/reduced over when fetching batches. For example,
            if `sample_dims = ['sequence', 'sample']` for variable with dimensions `['sequence', 'length', 'sample']`
            then a batch of data will have dimensions `['batch', 'length']`.
        variables : list[str]
            Which variables to sample from.
        transforms : Dict[str, (ndarray) -> Tensor], optional
            Transforms to apply to each variable. Defaults to transforming arrays to Tensor.

        For other parameters, see documentation for [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

        Returns
        -------
        DataLoader
        """
        if not TORCH_AVAILABLE:
            raise ImportError("Install PyTorch to get DataLoader from SeqData.")

        if transforms is None:
            _transforms: Dict[str, Callable[[NDArray], "torch.Tensor"]] = {}
        else:
            _transforms = transforms

        variables_not_in_ds = set(variables) - set(self.ds.data_vars.keys())
        if variables_not_in_ds:
            raise ValueError(
                f"Got variables that are not in the SeqData: {variables_not_in_ds}"
            )

        transform_vars_not_in_vars = set(_transforms.keys()) - set(variables)
        if transform_vars_not_in_vars:
            raise ValueError(
                f"Got transforms for variables that are not requested: {transform_vars_not_in_vars}"
            )

        transform_vars_not_in_ds = set(_transforms.keys()) - set(
            self.ds.data_vars.keys()
        )
        if transform_vars_not_in_ds:
            raise ValueError(
                f"Got transforms for variables that are not in the SeqData: {transform_vars_not_in_ds}"
            )

        dim_sizes = [self.ds.dims[d] for d in sample_dims]
        dataset = cartesian_product([np.arange(d) for d in dim_sizes])

        def collate_fn(indices: List[NDArray]):
            idx = np.vstack(indices)
            selector = {
                d: xr.DataArray(idx[:, i], dims="batch")
                for i, d in enumerate(sample_dims)
            }
            out = {
                k: arr.isel(selector, missing_dims="ignore").to_numpy()
                for k, arr in self.ds[variables].data_vars.items()
            }
            out_tensors = {
                k: _transforms.get(k, lambda x: torch.as_tensor(x))(arr)
                for k, arr in out.items()
            }
            return out_tensors

        return DataLoader(
            dataset,  # type: ignore
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
