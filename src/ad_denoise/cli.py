from dataclasses import dataclass
from os import getcwd
from pathlib import Path

import click
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from ad_denoise.lightning_modules import LightningModuleConfig
from ad_denoise.utils import load_config

from ._version_git import __version__


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, message="%(version)s")
@click.pass_context
def main(ctx: click.Context) -> None:
    """The main command line interface entry point.

    Args:
        ctx: The click context.
    """
    if ctx.invoked_subcommand is None:
        click.echo(main.get_help(ctx))


@dataclass
class TrainConfig:
    """A configuration schema for network training."""

    name: str
    model: LightningModuleConfig
    max_epochs: int


@main.command(help="Train a model on the given datasets")
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
def train(config_file: Path) -> None:  # noqa: D103
    config = load_config(config_file, TrainConfig)
    logger = TensorBoardLogger(str(Path(getcwd()).joinpath("logs")), config.name)
    trainer = Trainer(
        max_epochs=config.max_epochs,
        log_every_n_steps=1,
        accelerator="auto",
        logger=logger,
    )
    trainer.fit(config.model())
