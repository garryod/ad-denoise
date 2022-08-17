from typing import Callable, Generic, TypeVar

from .utils import SizedDataset

FrameInT = TypeVar("FrameInT")
FrameOutT = TypeVar("FrameOutT")


class ComputedFramesDataset(Generic[FrameInT, FrameOutT], SizedDataset[FrameOutT]):
    """A pytorch dataset which allows for computation on tensors."""

    def __init__(
        self,
        dataset: SizedDataset[FrameInT],
        computation: Callable[[FrameInT], FrameOutT],
    ) -> None:
        """Creates a pytorch dataset which allows for computation on tensors.

        Args:
            dataset: A sequence of datasets which loads a structure containing multiple
                tensors.
            computation: The computation to be performed on the tensors.
        """
        self.dataset = dataset
        self.computation = computation

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> FrameOutT:
        return self.computation(self.dataset[idx])
