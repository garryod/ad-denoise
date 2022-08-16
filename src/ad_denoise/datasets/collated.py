import operator
from itertools import accumulate, chain
from math import prod
from typing import Protocol, Sized

from more_itertools import take
from torch import Tensor
from torch.utils.data import Dataset


class SizedTensorsDataset(Sized, Protocol):
    """A protocol representing sized datasets which fetch a tuple of tensors."""

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        ...


class ZippedDatasets(Dataset):
    """A pytorch dataset which loads index aligned frames from hdf5 datasets.

    A pytorch dataset which loads index aligned frames from two iterables of hdf5
    datasets. This dataset creates a one to one mapping between frames in the left and
    right datasets, as such it requires the total number of frames present in each
    iterable of hdf5 datasets to be equal.
    """

    def __init__(self, *datasets: SizedTensorsDataset) -> None:
        """Creates a pytorch dataset which reads index matched data from hdf5 datasets.

        Args:
            datasets: A sequence of datasets of equal length.
        """
        self.datasets = datasets
        if any(len(datasets[0]) != len(dataset) for dataset in datasets):
            raise ValueError("All datasets must contain the same number of frames.")

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, idx: int) -> tuple[tuple[Tensor, ...], ...]:
        if idx >= len(self):
            raise IndexError
        return tuple(dataset[idx] for dataset in self.datasets)


class CrossedDatasets(Dataset):
    """A pytorch dataset which loads crossed frames from hdf5 datasets.

    A pytorch dataset which loads crossed frames from multiple sized datasets. The
    dataset creates a full crossing of loaded frames, resulting in a number of
    available frame sets equal to the product of the number of frames in the each
    dataset.
    """

    def __init__(self, *datasets: SizedTensorsDataset) -> None:
        """Creates a pytorch dataset which reads index crossed from hdf5 datasets.

        Args:
            datasets: A sequence of datasets of equal length.
        """
        self.datasets = datasets
        self.edges: list[int] = list(
            chain(
                (1,),
                take(
                    len(self.datasets) - 1,
                    accumulate(
                        (len(dataset) for dataset in self.datasets),
                        operator.mul,
                    ),
                ),
            )
        )

    def __len__(self) -> int:
        return prod(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx: int) -> tuple[tuple[Tensor, ...], ...]:
        if idx >= len(self):
            raise IndexError
        return tuple(
            dataset[(idx // edge) % len(dataset)]
            for dataset, edge in zip(self.datasets, self.edges)
        )
