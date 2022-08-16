import operator
from itertools import accumulate, chain
from math import prod

from more_itertools import take
from torch.utils.data import Dataset

from .utils import SizedTensorsDataset, Tensors


class ZippedDatasets(Dataset):
    """A pytorch dataset which loads index aligned frames from hdf5 datasets.

    A pytorch dataset which loads index aligned frames from two iterables of hdf5
    datasets. This dataset creates a one to one mapping between frames in each of the
    contained datasets. By default, the lengths of each dataset will be checked and an
    error raised if they are not equal.
    """

    def __init__(
        self, *datasets: SizedTensorsDataset, check_lengths: bool = True
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

    def __getitem__(self, idx: int) -> tuple[Tensors, ...]:
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

    def __getitem__(self, idx: int) -> tuple[Tensors, ...]:
        if idx >= len(self):
            raise IndexError
        return tuple(
            dataset[(idx // edge) % len(dataset)]
            for dataset, edge in zip(self.datasets, self.edges)
        )
