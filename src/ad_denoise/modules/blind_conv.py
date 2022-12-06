from torch import Tensor, no_grad, rand
from torch.nn import Module
from torch.nn.functional import conv2d
from torch.nn.parameter import Parameter


class BlindConv2D(Module):
    """A pytorch module which convolves learned blind spot kernel in 2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        half_width: int,
    ) -> None:
        """Creates a module which convolves learned kernel in 2d.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            half_width: The half width of the kernel, such that the full width is
                2*half_width+1.
        """
        super().__init__()
        if not half_width > 0:
            raise ValueError("Kernel half width must be positive.")

        self.half_width = half_width
        self.weights = Parameter(
            rand((out_channels, in_channels, 2 * half_width + 1, 2 * half_width + 1)),
            requires_grad=True,
        )

    def forward(self, input: Tensor):  # noqa: D102
        kernel = self.weights
        with no_grad():
            kernel[:, :, self.half_width, self.half_width] = 0
        return conv2d(input, kernel)
