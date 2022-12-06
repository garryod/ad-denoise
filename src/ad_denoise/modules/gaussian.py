from math import pi, sqrt
from typing import Optional

from torch import Tensor, as_tensor, exp, linspace, meshgrid, norm, rand, stack
from torch.nn import Module
from torch.nn.functional import conv2d
from torch.nn.parameter import Parameter


class GaussianKernel2D(Module):
    """A pytorch module which convolves a gaussian kernel with learned stdev in 2d."""

    def __init__(self, half_width: int, stdev: Optional[float] = None) -> None:
        """Creates a module which convolves a gaussian kernel with learned stdev in 2d.

        Args:
            half_width: The half width of the kernel to be created. Giving a kernel of
                shape: (2 * half_width + 1, 2 * half_width + 1).
            stdev: The initial standard deviation of kernel, if None a random value is
                selected from the uniform distribtuon on the interval [0,1). Defaults
                to None.
        """
        super().__init__()
        if not half_width > 0:
            raise ValueError("Kernel half width must be positive.")

        linvec = linspace(-half_width, half_width, 2 * half_width + 1)
        self.radii = Parameter(
            norm(stack(meshgrid(linvec, linvec, indexing="ij")), dim=0),
            requires_grad=False,
        )
        self.stdev = Parameter(
            as_tensor(stdev) if stdev is not None else rand((1,)), requires_grad=True
        )

    def _build_kernel(self) -> Tensor:
        return (
            (
                1
                / (self.stdev * sqrt(2 * pi))
                * exp(-0.5 * (self.radii / self.stdev) ** 2)
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return conv2d(x, self._build_kernel().type_as(x))


class BlindGaussianKernel2D(GaussianKernel2D):
    """A pytorch module which convolves a gaussian kernel with learned stdev in 2d."""

    def _build_kernel(self) -> Tensor:
        kernel = super()._build_kernel()
        kernel[:, :, kernel.shape[2] // 2, kernel.shape[3] // 2] = 0
        return kernel
