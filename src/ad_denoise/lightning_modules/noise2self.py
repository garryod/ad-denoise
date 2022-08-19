from dataclasses import dataclass

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module, Sequential, ZeroPad2d
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from ad_denoise.datasets.config import SizedDatasetConfig
from ad_denoise.modules import ScalarMultiply
from ad_denoise.modules.config import ModuleConfig
from ad_denoise.modules.gaussian import GaussianKernel2D

from .config import LightningModuleConfig


class Noise2Self(LightningModule):
    """A ligntning module which trains a nieve scaled gaussian denoiser.

    A ligntning module which trains a nieve scaled gaussian denoiser. This model will
    always converge close to a pair of weights which maximises the contribution by the
    centermost pixel.
    """

    def __init__(
        self,
        network: Module,
        train_dataset: Dataset[Tensor],
        val_dataset: Dataset[tuple[Tensor, Tensor]],
    ) -> None:
        """Creates a ligntning module which trains a nieve scaled gaussian denoiser.

        Args:
            network: The neural network to be trained using the noise2self methodology,
                this network must take an image in and return an equivilent image.
                Typically such models will attend to a doughnut around the predicted
                pixel as to avoid learning the identity function.
            train_dataset: A dataset which produces the training data, in the form of a
                single two dimensional tensor per index, which represents noisy data.
            val_dataset: A dataset which produces the evaluation data, in the form of a
                tuple containing two two dimensional tensors per index, the first of
                which represents noisy data and the second which represents clean data.
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.network = network

    def forward(self, x: Tensor) -> Tensor:  # type: ignore  # noqa: D102
        return self.network(x)

    def training_step(  # type: ignore
        self, batch: Tensor, batch_idx: int
    ) -> Tensor:  # noqa: D102
        inputs = targets = batch
        outputs = self.forward(inputs)
        loss = mse_loss(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(  # type: ignore
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:  # noqa: D102
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = mse_loss(outputs, targets)
        self.log("val_loss", loss)
        return loss

    def train_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            self.train_dataset, batch_size=32, shuffle=True, num_workers=12
        )

    def val_dataloader(self) -> DataLoader:  # noqa: D102
        return DataLoader(
            self.val_dataset, batch_size=32, shuffle=False, num_workers=12
        )

    def configure_optimizers(self) -> Adam:  # noqa: D102
        return Adam(self.parameters(), 0.1)


@dataclass
class ScaledGaussianConfig(ModuleConfig):
    """A configuration schema for a scaled gaussian network with padding."""

    __alias__ = "Gaussian"
    kernel_half_width: int

    def __call__(self) -> Module:  # noqa:D102
        return Sequential(
            ZeroPad2d(self.kernel_half_width),
            GaussianKernel2D(self.kernel_half_width),
            ScalarMultiply(),
        )


@dataclass
class Noise2SelfConfig(LightningModuleConfig):
    """A configuration schema for the noise2self method."""

    __alias__ = "Noise2Self"
    network: ModuleConfig
    train_dataset: SizedDatasetConfig[Tensor]
    val_dataset: SizedDatasetConfig[tuple[Tensor, Tensor]]

    def __call__(self) -> LightningModule:  # noqa: D102
        return Noise2Self(self.network(), self.train_dataset(), self.val_dataset())
