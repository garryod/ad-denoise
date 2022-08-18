from typing import NewType, Sequence, Sized, TypeVar

from torch.utils.data import Dataset as TorchDataset

#: The dimensionality of a frame.
Dim = NewType("Dim", int)
#: A sequence of frame dimensions.
Dims = Sequence[Dim]

T_co = TypeVar("T_co", covariant=True)


class SizedDataset(TorchDataset[T_co], Sized):
    """An abstract class representing a sized pytorch dataset."""
