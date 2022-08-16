from torch import float32, from_numpy
from torch.utils.data import Dataset

from .utils import (
    Dims,
    H5Keys,
    H5Paths,
    Tensors,
    get_dataset_edges,
    open_datasets,
    read_frame_datasets,
)


class SimpleHdf5(Dataset):
    """A pytorch dataset which loads frames at keys from multiple hdf5 paths."""

    def __init__(
        self,
        paths: H5Paths,
        keys: H5Keys,
        dims: Dims,
    ) -> None:
        """Creates a dataset which reads frames at keys from multiple hdf5 paths.

        Args:
            paths: A sequence of hdf5 file paths, from which data can be read.
            keys: A sequence of keys, pointing to datasets in each file, from which
                frames can be read.
            dims: A sequence of tensor dimensionalities, assumed to be trailing axis in
                the datasets.
        """
        self.dims = dims
        self.datasets = open_datasets(paths, keys)
        datasets_edges = tuple(
            get_dataset_edges(datasets, dims)
            for datasets, dims in zip(self.datasets, self.dims)
        )
        if not all(
            datasets_edges[0] == dataset_edges for dataset_edges in datasets_edges
        ):
            raise ValueError(
                "All datasets must contain the same number of samples from each file."
            )
        self.edges = datasets_edges[0]

    def __len__(self) -> int:
        return self.edges[-1]

    def __getitem__(self, idx: int) -> Tensors:
        return tuple(
            from_numpy(read_frame_datasets(datasets, idx, dims, self.edges))
            .unsqueeze(0)
            .type(float32)
            for datasets, dims in zip(self.datasets, self.dims)
        )
