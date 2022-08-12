from itertools import accumulate
from math import prod
from pathlib import Path
from typing import Iterable, NewType

import hdf5plugin  # noqa: F401
from h5py import Dataset, File
from numpy import ndarray, unravel_index

#: The path to an hdf5 file.
H5Path = NewType("H5Path", Path)
#: The key within an hdf5 file.
H5Key = NewType("H5Key", str)
#: A reference to a hdf5 dataset by path and key.
H5Dataset = tuple[H5Path, H5Key]
#: An iterable of references to hdf5 datasets by path and key.
H5Datasets = Iterable[H5Dataset]


def open_dataset(dataset: H5Dataset) -> Dataset:
    """Opens a hdf5 dataset by path and key.

    Args:
        dataset: A reference to a hdf5 dataset by path and key.

    Returns:
        Dataset: A readable hdf5 dataset object.
    """
    possibly_dataset = File(dataset[0])[dataset[1]]
    assert isinstance(possibly_dataset, Dataset)
    return possibly_dataset


def open_datasets(datasets: H5Datasets) -> list[Dataset]:
    """Opens hdf5 datasets by their paths and keys.

    Args:
        datasets: An iterable of references to hdf5 datasets by path and key.

    Returns:
        list[Dataset]: A list of readable hdf5 dataset objects.
    """
    return [open_dataset(dataset) for dataset in datasets]


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
