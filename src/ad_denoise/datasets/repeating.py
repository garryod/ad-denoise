from typing import Optional, TypeVar

from torch.utils.data import Dataset

from .utils import SizedDataset

T_co = TypeVar("T_co", covariant=True)


class RepeatingDataset(SizedDataset[T_co]):
    """A pytorch dataset which repeats the read data a given number of times."""

    def __init__(
        self,
        dataset: Dataset[T_co],
        apparent_length: int,
        child_index: int = 0,
    ) -> None:
        """Creates a pytorch dataset which repeats the read data.

        Args:
            dataset: A dataset from which to retrieve the data.
            apparent_length: The length which this dataset should report.
            child_index: The index from which data should be retrieved from the wrapped
                dataset. Defaults to 0.
        """
        self.dataset = dataset
        self.apparent_length = apparent_length
        self.child_index = child_index
        self._cached_data_store: Optional[T_co] = None

    def __len__(self) -> int:
        return self.apparent_length

    @property
    def _cached_data(self) -> T_co:
        if self._cached_data_store is None:
            self._cached_data_store = self.dataset[self.child_index]
        return self._cached_data_store

    def __getitem__(self, index: int) -> T_co:
        if index > len(self):
            raise IndexError
        return self._cached_data
