from torch import Tensor
from torch.utils.data import Dataset

from .single_frames import SingleFrames
from .utils import H5Key, H5Paths


class MatchedFramePairs(Dataset):
    """A pytorch dataset which loads matched pairs of frames from hdf5 datasets.

    A pytorch dataset which loads matched pairs of frames from two iterables of hdf5
    datasets. This dataset creates a one to one mapping between frames in the left and
    right datasets, as such it requires the total number of frames present in each
    iterable of hdf5 datasets to be equal.
    """

    def __init__(
        self,
        left_paths: H5Paths,
        right_paths: H5Paths,
        left_frames_key: H5Key,
        right_frames_key: H5Key,
        left_frame_dims: int = 2,
        right_frame_dims: int = 2,
    ) -> None:
        """Creates a pytorch dataset which reads matched pairs from hdf5 datasets.

        Args:
            left_paths: An iterable of hdf5 file paths from which data can be read.
            right_paths: An iterable of hdf5 file paths from which data can be read.
            left_frames_key: The key of the dataset to be read.
            right_frames_key: The key of the dataset to be read.
            left_frame_dims: The trailing dimensionality of the frame. Defaults to 2.
            right_frame_dims: The trailing dimensionality of the frame. Defaults to 2.
        """
        self.left_dataset = SingleFrames(left_paths, left_frames_key, left_frame_dims)
        self.right_dataset = SingleFrames(
            right_paths, right_frames_key, right_frame_dims
        )
        if len(self.left_dataset) != len(self.right_dataset):
            raise ValueError(
                "Left & right datasets must contain the same number of frames."
            )

    def __len__(self) -> int:
        return len(self.left_dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.left_dataset[idx], self.right_dataset[idx]


class CrossedFramePairs(Dataset):
    """A pytorch dataset which loads crossed pairs of frames from hdf5 datasets.

    A pytorch dataset which loads matched pairs of frames from two iterables of hdf5
    datasets. The dataset creates a full crossing of loaded pairs, resulting in a
    number of available pairs equal to the product of the number of frames in the left
    and right datasets.
    """

    def __init__(
        self,
        left_paths: H5Paths,
        right_paths: H5Paths,
        left_frames_key: H5Key,
        right_frames_key: H5Key,
        left_frame_dims: int = 2,
        right_frame_dims: int = 2,
    ) -> None:
        """Creates a pytorch dataset which reads crossed pairs from hdf5 datasets.

        Args:
            left_paths: An iterable of hdf5 file paths from which data can be read.
            right_paths: An iterable of hdf5 file paths from which data can be read.
            left_frames_key: The key of the dataset to be read.
            right_frames_key: The key of the dataset to be read.
            left_frame_dims: The trailing dimensionality of the frame. Defaults to 2.
            right_frame_dims: The trailing dimensionality of the frame. Defaults to 2.
        """
        self.left_dataset = SingleFrames(left_paths, left_frames_key, left_frame_dims)
        self.right_dataset = SingleFrames(
            right_paths, right_frames_key, right_frame_dims
        )

    def __len__(self) -> int:
        return len(self.left_dataset) * len(self.right_dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        left_idx = idx // len(self.left_dataset)
        right_idx = idx % len(self.left_dataset)
        return self.left_dataset[left_idx], self.right_dataset[right_idx]
