from dataclasses import dataclass
from pathlib import Path

import click
from pytorch_lightning import Trainer

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

    model: LightningModuleConfig
    max_epochs: int


@main.command(help="Train a model on the given datasets")
@click.argument("config_file", type=click.Path(exists=True, dir_okay=False))
def train(config_file: Path) -> None:  # noqa: D103
    config = load_config(config_file, TrainConfig)
    trainer = Trainer(max_epochs=config.max_epochs, log_every_n_steps=1)
    trainer.fit(config.model())
