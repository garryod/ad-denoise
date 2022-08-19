from math import isclose
from unittest.mock import MagicMock, patch

import pytest
from more_itertools import ilen
from torch import Tensor
from torch import isclose as isclose_tensor
from torch import tensor

from ad_denoise.modules.scalar_multiply import ScalarMultiply


@pytest.mark.parametrize(
    ("input", "scalar", "expected"), [(tensor([0, 1, 0]), 2.0, tensor([0, 2.0, 0]))]
)
def test_scalar_multiply_multiplies_by_scalar(
    input: Tensor, scalar: float, expected: Tensor
):
    module = ScalarMultiply(scalar)
    assert isclose_tensor(expected, module(input)).all()


def test_scalar_multiply_has_weight():
    assert 1 == ilen(ScalarMultiply().parameters())


def test_scalar_multiply_generates_random_scalar():
    mock_rand = MagicMock(return_value=tensor([3.14]))
    with patch("ad_denoise.modules.scalar_multiply.rand", mock_rand):
        module = ScalarMultiply()
        assert isclose(3.14, module.scalar.item(), rel_tol=1e-6)
