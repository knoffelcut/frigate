import logging

from .identifier_config import (  # noqa: F401
    InputTensorEnum,
    ModelConfig,
    PixelFormatEnum,
)
from .identifier_types import (  # noqa: F401
    IdentifierConfig,
    IdentifierTypeEnum,
    api_types,
)

logger = logging.getLogger(__name__)


def create_identifier(identifier_config):
    api = api_types.get(identifier_config.type)
    if not api:
        raise ValueError(identifier_config.type)
    return api(identifier_config)
