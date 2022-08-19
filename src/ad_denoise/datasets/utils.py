from typing import NewType, Sized, TypeVar

from torch.utils.data import Dataset as TorchDataset

#: The dimensionality of a frame.
Dim = NewType("Dim", int)

T_co = TypeVar("T_co", covariant=True)


class SizedDataset(TorchDataset[T_co], Sized):
    """An abstract class representing a sized pytorch dataset."""
