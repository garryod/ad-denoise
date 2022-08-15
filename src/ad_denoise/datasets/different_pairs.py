from torch import Tensor, float32, from_numpy
from torch.utils.data import Dataset

from .utils import H5Datasets, get_dataset_edges, open_datasets, read_frame_datasets


class MatchedFramePairs(Dataset):
    """A pytorch dataset which loads matched pairs of frames from hdf5 datasets.

    A pytorch dataset which loads matched pairs of frames from two iterables of hdf5
    datasets. This dataset creates a one to one mapping between frames in the left and
    right datasets, as such it requires the total number of frames present in each
    iterable of hdf5 datasets to be equal.
    """

    def __init__(
        self,
        left_datasets: H5Datasets,
        right_datasets: H5Datasets,
        left_frame_dims: int = 2,
        right_frame_dims: int = 2,
    ) -> None:
        """Creates a pytorch dataset which reads matched pairs from hdf5 datasets.

        Args:
            left_datasets: An iterable of hdf5 file paths and the key of the dataset to
                be read.
            right_datasets: An iterable of hdf5 file paths and the key of the dataset to
                be read.
            left_frame_dims: The trailing dimensionality of the frame. Defaults to 2.
            right_frame_dims: The trailing dimensionality of the frame. Defaults to 2.
        """
        self.left_frame_dims = left_frame_dims
        self.right_frame_dims = right_frame_dims
        self.left_datasets = open_datasets(left_datasets)
        self.right_datasets = open_datasets(right_datasets)
        self.left_edges = get_dataset_edges(self.left_datasets, self.left_frame_dims)
        self.right_edges = get_dataset_edges(self.right_datasets, self.right_frame_dims)
        if self.left_edges[-1] != self.right_edges[-1]:
            raise ValueError(
                "Left & right datasets must contain the same number of frames."
            )

    def __len__(self) -> int:
        return self.left_edges[-1]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        left_frame = (
            from_numpy(
                read_frame_datasets(
                    self.left_datasets, idx, self.left_frame_dims, self.left_edges
                )
            )
            .unsqueeze(0)
            .type(float32)
        )
        right_frame = (
            from_numpy(
                read_frame_datasets(
                    self.right_datasets, idx, self.right_frame_dims, self.right_edges
                )
            )
            .unsqueeze(0)
            .type(float32)
        )
        return left_frame, right_frame


class CrossedFramePairs(Dataset):
    """A pytorch dataset which loads crossed pairs of frames from hdf5 datasets.

    A pytorch dataset which loads matched pairs of frames from two iterables of hdf5
    datasets. The dataset creates a full crossing of loaded pairs, resulting in a
    number of available pairs equal to the product of the number of frames in the left
    and right datasets.
    """

    def __init__(
        self,
        left_datasets: H5Datasets,
        right_datasets: H5Datasets,
        left_frame_dims: int = 2,
        right_frame_dims: int = 2,
    ) -> None:
        """Creates a pytorch dataset which reads crossed pairs from hdf5 datasets.

        Args:
            left_datasets: An iterable of hdf5 file paths and the key of the dataset to
                be read.
            right_datasets: An iterable of hdf5 file paths and the key of the dataset to
                be read.
            left_frame_dims: The trailing dimensionality of the frame. Defaults to 2.
            right_frame_dims: The trailing dimensionality of the frame. Defaults to 2.
        """
        super().__init__()
        self.left_frame_dims = left_frame_dims
        self.right_frame_dims = right_frame_dims
        self.left_datasets = open_datasets(left_datasets)
        self.right_datasets = open_datasets(right_datasets)
        self.left_edges = get_dataset_edges(self.left_datasets, self.left_frame_dims)
        self.right_edges = get_dataset_edges(self.right_datasets, self.right_frame_dims)

    def __len__(self) -> int:
        return self.left_edges[-1] * self.right_edges[-1]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        left_idx = idx // self.left_edges[-1]
        right_idx = idx % self.left_edges[-1]

        left_frame = (
            from_numpy(
                read_frame_datasets(
                    self.left_datasets, left_idx, self.left_frame_dims, self.left_edges
                )
            )
            .unsqueeze(0)
            .type(float32)
        )
        right_frame = (
            from_numpy(
                read_frame_datasets(
                    self.right_datasets,
                    right_idx,
                    self.right_frame_dims,
                    self.right_edges,
                )
            )
            .unsqueeze(0)
            .type(float32)
        )
        return left_frame, right_frame
