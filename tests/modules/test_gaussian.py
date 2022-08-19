from math import isclose
from unittest.mock import MagicMock, patch

import pytest
from more_itertools import ilen
from torch import tensor, zeros

from ad_denoise.modules.gaussian import GaussianKernel2D


@pytest.mark.parametrize(
    ("location", "stdev", "expected"),
    [
        ((0, 0), 0.1, 3.98942),
        ((1, 0), 1.7, 0.197389),
        ((1, 1), 1.7, 0.166030),
        ((0, 2), 1.7, 0.117466),
    ],
)
def test_gaussian_kernel_2d_multiplies_by_gaussian(
    location: tuple[int, int], stdev: float, expected: float
):
    size = max(*location, 1)
    model = GaussianKernel2D(size, stdev)
    input = zeros((1, 2 * size + 1, 2 * size + 1))
    input[0, location[0] + size, location[1] + size] = 1
    assert isclose(expected, model.forward(input).item(), rel_tol=1e-5)


def test_gaussian_kernel_has_weight():
    assert 1 == ilen(GaussianKernel2D(42).parameters())


def test_gaussian_kernel_generates_random_stdev():
    mock_rand = MagicMock(return_value=tensor([3.14]))
    with patch("ad_denoise.modules.gaussian.rand", mock_rand):
        module = GaussianKernel2D(42)
        assert isclose(3.14, module.stdev.item(), rel_tol=1e-6)
