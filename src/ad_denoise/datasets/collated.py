import operator
from dataclasses import dataclass
from itertools import accumulate, chain
from math import prod
from typing import Any, TypeVar, cast

from more_itertools import take

from .config import SizedDatasetConfig
from .utils import SizedDataset


class ZippedDatasets(SizedDataset[tuple[Any, ...]]):
    """A pytorch dataset which loads index aligned frames from hdf5 datasets.

    A pytorch dataset which loads index aligned frames from two iterables of hdf5
    datasets. This dataset creates a one to one mapping between frames in each of the
    contained datasets. By default, the lengths of each dataset will be checked and an
    error raised if they are not equal.
    """

    def __init__(
        self, *datasets: SizedDataset[Any], check_lengths: bool = True
    ) -> None:
        """Creates a pytorch dataset which reads index matched data from hdf5 datasets.

        Args:
            datasets: A sequence of datasets of equal length.
            check_lengths: If True, a value error will be raised if datasets are not of
                equal length. Defaults to True.
        """
        self.datasets = datasets
        if check_lengths and any(
            len(datasets[0]) != len(dataset) for dataset in datasets
        ):
            raise ValueError("All datasets must contain the same number of frames.")

    def __len__(self) -> int:
        return min(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        if idx >= len(self):
            raise IndexError
        return tuple(dataset[idx] for dataset in self.datasets)


@dataclass
class ZippedDatasetsConfig(SizedDatasetConfig[tuple[Any, ...]]):
    """A configuration schema for zipped datasets."""

    __alias__ = "ZippedDatasets"
    datasets: list[SizedDatasetConfig]
    check_lengths: bool

    def __call__(self) -> SizedDataset[tuple[Any, ...]]:  # noqa: D102
        return ZippedDatasets(
            *(dataset() for dataset in self.datasets), check_lengths=self.check_lengths
        )


T1 = TypeVar("T1")
T2 = TypeVar("T2")


class InputTargetDataset(SizedDataset[tuple[T1, T2]]):
    """A pytorch dataset containing a zipped input and target dataset."""

    def __init__(
        self,
        input: SizedDataset[T1],
        target: SizedDataset[T2],
        check_lengths: bool = True,
    ) -> None:
        """Creates a pytorch dataset with a zipped input and target dataset.

        Args:
            input: The input dataset.
            target: The target dataset.
            check_lengths: If True, a value error will be raised if datasets are not of
                equal length. Defaults to True.
        """
        self.dataset = ZippedDatasets(input, target, check_lengths=check_lengths)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[T1, T2]:
        return cast(tuple[T1, T2], self.dataset[idx])


@dataclass
class InputTargetDatasetConfig(SizedDatasetConfig[tuple[T1, T2]]):
    """A configuration schema for input target datasets."""

    __alias__ = "InputTargetDataset"
    input: SizedDatasetConfig[T1]
    target: SizedDatasetConfig[T2]

    def __call__(self) -> SizedDataset[tuple[T1, T2]]:  # noqa: D102
        return InputTargetDataset(self.input(), self.target())


class CrossedDatasets(SizedDataset[tuple[Any, ...]]):
    """A pytorch dataset which loads crossed frames from hdf5 datasets.

    A pytorch dataset which loads crossed frames from multiple sized datasets. The
    dataset creates a full crossing of loaded frames, resulting in a number of
    available frame sets equal to the product of the number of frames in the each
    dataset.
    """

    def __init__(self, *datasets: SizedDataset[Any]) -> None:
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

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        if idx >= len(self):
            raise IndexError
        return tuple(
            dataset[(idx // edge) % len(dataset)]
            for dataset, edge in zip(self.datasets, self.edges)
        )


@dataclass
class CrossedDatasetsConfig(SizedDatasetConfig[tuple[Any, ...]]):
    """A configuration schema for crossed datasets."""

    __alias__ = "CrossedDatasets"
    datasets: list[SizedDatasetConfig]

    def __call__(self) -> SizedDataset[tuple[Any, ...]]:  # noqa: D102
        return CrossedDatasets(*(dataset() for dataset in self.datasets))
