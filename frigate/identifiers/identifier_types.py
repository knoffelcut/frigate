import importlib
import logging
import pkgutil
from enum import Enum
from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from . import plugins
from .identifier_api import IdentificationApi
from .identifier_config import BaseIdentifierConfig

logger = logging.getLogger(__name__)


_included_modules = pkgutil.iter_modules(plugins.__path__, plugins.__name__ + ".")

plugin_modules = []

for _, name, _ in _included_modules:
    try:
        # currently openvino may fail when importing
        # on an arm device with 64 KiB page size.
        plugin_modules.append(importlib.import_module(name))
    except ImportError as e:
        logger.error(f"Error importing identifier runtime: {e}")


api_types = {det.type_key: det for det in IdentificationApi.__subclasses__()}


class StrEnum(str, Enum):
    pass


IdentifierTypeEnum = StrEnum("IdentifierTypeEnum", {k: k for k in api_types})

IdentifierConfig = Annotated[
    Union[tuple(BaseIdentifierConfig.__subclasses__())],
    Field(discriminator="type"),
]
