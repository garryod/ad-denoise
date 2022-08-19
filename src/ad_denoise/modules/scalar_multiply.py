from typing import Optional

from torch import Tensor, rand, tensor
from torch.nn import Module
from torch.nn.parameter import Parameter


class ScalarMultiply(Module):
    """A pytorch module which multiplies the input tensor by a learned scalar."""

    def __init__(self, scalar: Optional[float] = None) -> None:
        """Creates a module which multiplies the input tensor by a learned scalar.

        Args:
            scalar: The initial scalar value, if None a random value is selected from
                the uniform distribution on the interval [0,1). Defaults to None.
        """
        super().__init__()
        self.scalar = Parameter(
            tensor(scalar) if scalar is not None else rand((1,)), requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return x * self.scalar
