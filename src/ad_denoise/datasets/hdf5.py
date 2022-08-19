from dataclasses import dataclass
from itertools import accumulate
from math import prod
from pathlib import Path
from typing import Iterable, NewType, Optional, Sequence

import hdf5plugin  # noqa: F401
from h5py import Dataset, File
from numpy import atleast_1d, ndarray, unravel_index
from torch import Tensor, float32, from_numpy

from .config import SizedDatasetConfig
from .utils import Dim, SizedDataset

#: The path to an hdf5 file.
H5Path = NewType("H5Path", Path)
#: A sequence of paths to hdf5 files.
H5Paths = Sequence[H5Path]
#: The key within an hdf5 file.
H5Key = NewType("H5Key", str)
#: A sequence of keys within an hdf5 file.
H5Keys = Sequence[H5Key]


class SimpleHdf5(SizedDataset[Tensor]):
    """A pytorch dataset which loads frames at keys from multiple hdf5 paths."""

    def __init__(
        self,
        paths: H5Paths,
        key: H5Key,
        dimensions: Dim,
    ) -> None:
        """Creates a dataset which reads frames at keys from multiple hdf5 paths.

        Args:
            paths: A sequence of hdf5 file paths, from which data can be read.
            key: A hdf5 key, pointing to dataset in each file, from which frames can be
                read.
            dimensions: The data dimensionality, assumed to be trailing axis in the
                dataset.
        """
        self.dimensions = dimensions
        self.datasets = SimpleHdf5.open_datasets(paths, key)
        self.edges = SimpleHdf5.get_dataset_edges(self.datasets, self.dimensions)

    @staticmethod
    def _open_files(paths: H5Paths) -> list[File]:
        """Opens hdf5 files by their paths.

        Args:
            paths: A sequence of paths to hdf5 files.

        Returns:
            list[File]: A list of opened hdf5 files.
        """
        return [File(path) for path in paths]

    @staticmethod
    def _get_dataset(file: File, key: H5Key) -> Dataset:
        possibly_dataset = file[key]
        assert isinstance(possibly_dataset, Dataset)
        return possibly_dataset

    @staticmethod
    def open_datasets(paths: H5Paths, key: H5Key) -> list[Dataset]:
        """Opens hdf5 datasets by their paths and keys.

        Args:
            paths: A sequence of paths to hdf5 files.
            key: The key within each hdf5 file.

        Returns:
            list[Dataset]: A list of opened hdf5 datasets.
        """
        return [
            SimpleHdf5._get_dataset(file, key) for file in SimpleHdf5._open_files(paths)
        ]

    @staticmethod
    def get_frame_count(dataset: Dataset, frame_dims: int) -> int:
        """Computes the number of frames in a dataset.

        Args:
            dataset: A readable hdf5 dataset object.
            frame_dims: The trailing dimensionality of the frame.

        Returns:
            int: The number of frames in the dataset.
        """
        return prod(dataset.shape[: len(dataset.shape) - frame_dims])

    @staticmethod
    def get_frame_counts(datasets: Iterable[Dataset], frame_dims: int) -> list[int]:
        """Computes the number of frames in each dataset.

        Args:
            datasets: An iterable of readable hdf5 dataset objects.
            frame_dims: The trailing dimensionality of the frame.

        Returns:
            list[int]: A list of the number of frames in each dataset.
        """
        return [SimpleHdf5.get_frame_count(dataset, frame_dims) for dataset in datasets]

    @staticmethod
    def get_dataset_edges(datasets: Iterable[Dataset], frame_dims: int) -> list[int]:
        """Computes the cumulative sum of the number of frames in each dataset.

        Args:
            datasets: An iterable of readable hdf5 dataset objects.
            frame_dims: The trailing dimensionality of the frame.

        Returns:
            list[int]: A list of the total number of frames in all preceeding datasets.
        """
        return [0, *accumulate(SimpleHdf5.get_frame_counts(datasets, frame_dims))]

    @staticmethod
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

    @staticmethod
    def read_frame(dataset: Dataset, idx: int, frame_dims: int) -> ndarray:
        """Reads a frame of dimensionality frame_dims from a dataset at idx.

        Args:
            dataset: A readable hdf5 dataset object.
            idx: The linearised index of a frame in the dataset.
            frame_dims: The trailing dimensionality of the frame.

        Returns:
            ndarray: An array of dimensionality frame_dims containing the frame data.
        """
        return atleast_1d(
            dataset[
                unravel_index(idx, dataset.shape[: len(dataset.shape) - frame_dims])
            ]
        )

    @staticmethod
    def read_frame_datasets(
        datasets: list[Dataset],
        idx: int,
        frame_dims: int,
        edges: Optional[list[int]] = None,
    ) -> ndarray:
        """Reads a frame of dimensionality frame_dims from a list of datasets at idx.

        Reads a frame of dimensionality frame_dims from a list of datasets at idx. A
        list of dataset edges, as computed by get_dataset_edges, may be supplied in
        order to avoid repeat computation of this value.

        Args:
            datasets: A list of readable hdf5 dataset objects.
            edges: A list of the total number of frames in all preceeding datasets.
            idx: The linearised index of a frame in the datasets.
            frame_dims: The trailing dimensionality of the frame.

        Returns:
            ndarray: An array of dimensionality frame_dims containing the frame data.
        """
        edges = (
            edges
            if edges is not None
            else SimpleHdf5.get_dataset_edges(datasets, frame_dims)
        )
        dataset_idx = SimpleHdf5.get_dataset_index(idx, edges)
        dataset = datasets[dataset_idx]
        start_idx = edges[dataset_idx]
        return SimpleHdf5.read_frame(dataset, idx - start_idx, frame_dims)

    def __len__(self) -> int:
        return self.edges[-1]

    def __getitem__(self, idx: int) -> Tensor:
        return (
            from_numpy(
                SimpleHdf5.read_frame_datasets(
                    self.datasets, idx, self.dimensions, self.edges
                )
            )
            .unsqueeze(0)
            .type(float32)
        )


@dataclass
class SimpleHdf5DatasetConfig(SizedDatasetConfig):
    """A configuration schema for a simple hdf5 dataset."""

    __alias__ = "SimpleHdf5Dataset"
    paths: list[H5Path]
    key: H5Key
    dimensions: Dim

    def __call__(self) -> SizedDataset[Tensor]:  # noqa: D102
        return SimpleHdf5(self.paths, self.key, self.dimensions)
