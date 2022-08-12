from itertools import accumulate
from math import prod
from pathlib import Path
from typing import Optional, cast

import hdf5plugin  # noqa: F401
from h5py import Dataset as H5Dataset
from h5py import File as H5File
from numpy import unravel_index
from torch import Tensor, float32, from_numpy
from torch.utils.data import Dataset


class SingleFrames(Dataset):
    """A pytorch dataset which loads individual frames from a set of hdf5 datasets."""

    def __init__(self, datasets: set[tuple[Path, str]], frame_dims: int = 2) -> None:
        """Creates a pytorch dataset which reads frames hdf5 datasets.

        Args:
            datasets: A set of hdf5 file paths and the key of the dataset to be read.
            frame_dims: The trailing dimensionality of the frame. Defaults to 2.
        """
        super().__init__()
        self.frame_dims = frame_dims
        self.datasets = [
            cast(H5Dataset, H5File(path)[dataset_key]) for path, dataset_key in datasets
        ]
        lengths = [prod(dataset.shape[: -self.frame_dims]) for dataset in self.datasets]
        self.cum_lengths = [0, *accumulate(lengths)]

    def _get_dataset_index(self, frame_idx: int) -> Optional[int]:
        for dataset_idx, cum_length in enumerate(self.cum_lengths):
            if frame_idx < cum_length:
                return dataset_idx - 1
        return None

    def __len__(self) -> int:
        return self.cum_lengths[-1]

    def __getitem__(self, idx: int) -> Tensor:
        dataset_idx = self._get_dataset_index(idx)
        if dataset_idx is None:
            raise IndexError()
        dataset = self.datasets[dataset_idx]
        start_idx = self.cum_lengths[dataset_idx]
        return (
            from_numpy(
                dataset[
                    unravel_index(idx - start_idx, dataset.shape[: -self.frame_dims])
                ]
            )
            .unsqueeze(0)
            .type(float32)
        )
