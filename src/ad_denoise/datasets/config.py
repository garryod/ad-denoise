from dataclasses import dataclass
from typing import Generic, TypeVar

from ad_denoise.utils import as_tagged_union

from .utils import SizedDataset

T = TypeVar("T")


@dataclass
@as_tagged_union
class SizedDatasetConfig(Generic[T]):
    """A configuration schema for sized datasets."""

    def __call__(self) -> SizedDataset[T]:  # noqa: D102
        raise NotImplementedError(self)
