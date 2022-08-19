from dataclasses import dataclass

from torch.nn import Module

from ad_denoise.utils import as_tagged_union


@dataclass
@as_tagged_union
class ModuleConfig:
    """A configuration schema for pytorch modules."""

    def __call__(self) -> Module:  # noqa: D102
        raise NotImplementedError(self)
