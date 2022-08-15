from itertools import accumulate
from math import prod
from pathlib import Path
from typing import Iterable, NewType, Optional, Sequence

import hdf5plugin  # noqa: F401
from h5py import Dataset, File
from numpy import ndarray, unravel_index

#: The path to an hdf5 file.
H5Path = NewType("H5Path", Path)
#: A sequence of paths to hdf5 files.
H5Paths = Sequence[H5Path]
#: The key within an hdf5 file.
H5Key = NewType("H5Key", str)
#: A sequence of keys within an hdf5 file.
H5Keys = Sequence[H5Key]


def _get_dataset(file: File, key: H5Key) -> Dataset:
    possibly_dataset = file[key]
    assert isinstance(possibly_dataset, Dataset)
    return possibly_dataset


def open_datasets(paths: H5Paths, keys: H5Keys) -> tuple[list[Dataset], ...]:
    """Opens hdf5 datasets by their paths and keys.

    Args:
        datasets: An iterable of references to hdf5 datasets by path and key.

    Returns:
        tuple[list[Dataset], ...]: A tuple across keys containing lists of readable
            hdf5 dataset objects.
    """
    files = [File(path) for path in paths]
    return tuple([_get_dataset(file, key) for file in files] for key in keys)


def get_frame_count(dataset: Dataset, frame_dims: int) -> int:
    """Computes the number of frames in a dataset.

    Args:
        dataset: A readable hdf5 dataset object.
        frame_dims: The trailing dimensionality of the frame.

    Returns:
        int: The number of frames in the dataset.
    """
    return prod(dataset.shape[:-frame_dims])


def get_frame_counts(datasets: Iterable[Dataset], frame_dims: int) -> list[int]:
    """Computes the number of frames in each dataset.

    Args:
        datasets: An iterable of readable hdf5 dataset objects.
        frame_dims: The trailing dimensionality of the frame.

    Returns:
        list[int]: A list of the number of frames in each dataset.
    """
    return [get_frame_count(dataset, frame_dims) for dataset in datasets]


def get_dataset_edges(datasets: Iterable[Dataset], frame_dims: int) -> list[int]:
    """Computes the cumulative sum of the number of frames in each dataset.

    Args:
        datasets: An iterable of readable hdf5 dataset objects.
        frame_dims: The trailing dimensionality of the frame.

    Returns:
        list[int]: A list of the total number of frames in all preceeding datasets.
    """
    return [0, *accumulate(get_frame_counts(datasets, frame_dims))]


def get_dataset_index(idx: int, edges: list[int]) -> int:
    """Computes the index of the dataset which contains the frame at idx.

    Args:
        frame_idx: The index of a frame which resides in a dataset bound by edges.
        edges: A list of the total number of frames in all preceeding datasets.

    Returns:
        int: The index of the dataset which contains the requested frame.
    """
    for dataset_idx, edge in enumerate(edges):
        if idx < edge:
            return dataset_idx - 1
    raise IndexError("Frame index out of bounds for given edges.")


def read_frame(dataset: Dataset, idx: int, frame_dims: int) -> ndarray:
    """Reads a frame of dimensionality frame_dims from a dataset at idx.

    Args:
        dataset: A readable hdf5 dataset object.
        idx: The linearised index of a frame in the dataset.
        frame_dims: The trailing dimensionality of the frame.

    Returns:
        ndarray: An array of dimensionality frame_dims containing the frame data.
    """
    return dataset[unravel_index(idx, dataset.shape[:-frame_dims])]


def read_frame_datasets(
    datasets: list[Dataset],
    idx: int,
    frame_dims: int,
    edges: Optional[list[int]] = None,
) -> ndarray:
    """Reads a frame of dimensionality frame_dims from a list of datasets at idx.

    Reads a frame of dimensionality frame_dims from a list of datasets at idx. A list
    of dataset edges, as computed by get_dataset_edges, may be supplied in order to
    avoid repeat computation of this value.

    Args:
        datasets: A list of readable hdf5 dataset objects.
        edges: A list of the total number of frames in all preceeding datasets.
        idx: The linearised index of a frame in the datasets.
        frame_dims: The trailing dimensionality of the frame.

    Returns:
        ndarray: An array of dimensionality frame_dims containing the frame data.
    """
    edges = edges if edges is not None else get_dataset_edges(datasets, frame_dims)
    dataset_idx = get_dataset_index(idx, edges)
    dataset = datasets[dataset_idx]
    start_idx = edges[dataset_idx]
    return read_frame(dataset, idx - start_idx, frame_dims)
