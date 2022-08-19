from dataclasses import dataclass

from pytorch_lightning import LightningModule

from ad_denoise.utils import as_tagged_union


@dataclass
@as_tagged_union
class LightningModuleConfig:
    """A configuration schema for pytorch lightning modules."""

    def __call__(self) -> LightningModule:  # noqa: D102
        raise NotImplementedError(self)
