from typing import Iterable

import pytest
from torch import Tensor

from ad_denoise.modules.shift import Shift


@pytest.mark.parametrize(
    ("input", "dim_shifts", "expected"),
    [
        (
            Tensor([[1, 2], [3, 4]]),
            [(0, 1)],
            Tensor([[0, 0], [1, 2]]),
        ),
        (
            Tensor([[1, 2], [3, 4]]),
            [(0, -1)],
            Tensor([[3, 4], [0, 0]]),
        ),
        (
            Tensor([[1, 2], [3, 4]]),
            [(1, 1)],
            Tensor([[0, 1], [0, 3]]),
        ),
        (
            Tensor([[1, 2], [3, 4]]),
            [(0, 1), (1, 1)],
            Tensor([[0, 0], [0, 1]]),
        ),
    ],
)
def test_shift(input: Tensor, dim_shifts: Iterable[tuple[int, int]], expected: Tensor):
    shift_module = Shift(dim_shifts)
    assert (expected == shift_module.forward(input)).all()
