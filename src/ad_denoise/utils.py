from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterator, Type, TypeVar

from apischema import deserialize, deserializer, serializer
from apischema.conversions import Conversion
from apischema.tagged_unions import Tagged, TaggedUnion, get_tagged
from yaml import Loader, load

# Implementation adapted from apischema example: Class as tagged union of its subclasses
# see: https://wyfo.github.io/apischema/examples/subclass_tagged_union/

T = TypeVar("T")
#: A class
Cls = TypeVar("Cls", bound=type)


def rec_subclasses(cls: type) -> Iterator[type]:
    """Recursive implementation of type.__subclasses__.

    Args:
        cls (Type): The base class.

    Returns:
        Iterator[type]: An iterator of subclasses.
    """
    for sub_cls in cls.__subclasses__():
        yield sub_cls
        yield from rec_subclasses(sub_cls)


#: Whether the current class is registered as a tagged union
is_tagged_union: Dict[Type[Any], bool] = DefaultDict(lambda: False)


def as_tagged_union(cls: Cls) -> Cls:  # noqa: D205
    """A decorator to make a config base class which can deserialize sub-classes.

    A decorator which makes a config class the root of a tagged union of sub-classes
    allowing for serialization and deserialization of config trees by class alias. The
    function registers both an apischema serialization and an apischema deserialization
    conversion for the base class which perform lookup based on a tagged union of
    aliased sub-classes.

    Args:
        cls (Cls): The config base class.

    Returns:
        Cls: The modified config base class.
    """

    def serialization() -> Conversion:
        annotations = {
            sub.__name__: Tagged[sub] for sub in rec_subclasses(cls)  # type: ignore
        }
        namespace = {"__annotations__": annotations}
        tagged_union = type(cls.__name__, (TaggedUnion,), namespace)
        return Conversion(
            lambda obj: tagged_union(**{obj.__class__.__name__: obj}),
            source=cls,
            target=tagged_union,
            inherited=False,
        )

    def deserialization() -> Conversion:
        annotations: Dict[str, Any] = {}
        namespace: Dict[str, Any] = {"__annotations__": annotations}
        for sub in rec_subclasses(cls):
            sub_name = getattr(sub, "__alias__", sub.__name__)
            annotations[sub_name] = Tagged[sub]  # type: ignore
        tagged_union = type(cls.__name__ + "TaggedUnion", (TaggedUnion,), namespace)
        return Conversion(
            lambda obj: get_tagged(obj)[1], source=tagged_union, target=cls
        )

    deserializer(lazy=deserialization, target=cls)
    serializer(lazy=serialization, source=cls)
    is_tagged_union[cls] = True
    return cls


def load_config(path: Path, config_type: Type[T]) -> T:
    """Load the configuation of config_type from a yaml file located at path.

    Args:
        path: A path to the configuration file.
        config_type: The type of the configuration object, used as a deserialization
            schema.

    Returns:
        T: The configuration object.
    """
    with open(path) as config_file:
        toml_dict = load(config_file, Loader=Loader)
        config = deserialize(config_type, toml_dict)
        return config
