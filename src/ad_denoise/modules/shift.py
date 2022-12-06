from typing import Iterable

from torch import Tensor, roll
from torch.nn import Module


class Shift(Module):
    """A pytorch module which translates a tensor, filling with zero.

    A pytorch module which translates a tensor by an arbitrary pixel value across
    multiple dimensions, filling all newly values with zero.
    """

    def __init__(self, dim_shifts: Iterable[tuple[int, int]]) -> None:
        """Creates a module which translates a tensor, filling with zero.

        Args:
            dim_shifts: An iterable, where each item is a tuple of the dimension on
                which the translation should occur and the number of pixels by which it
                should be translated.
        """
        super().__init__()
        self.shifts = tuple(shift for _, shift in dim_shifts)
        self.dims = tuple(dim for dim, _ in dim_shifts)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x = roll(x, self.shifts, self.dims)
        for shift, dim in zip(self.shifts, self.dims):
            if shift > 0:
                x.tensor_split(shift + 1, dim)[0].zero_()
            elif shift < 0:
                x.tensor_split(x.shape[dim] - shift, dim)[1].zero_()
        return x
