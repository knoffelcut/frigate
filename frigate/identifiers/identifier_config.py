import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Extra, Field

import frigate.detectors.detector_config

logger = logging.getLogger(__name__)


class PixelFormatEnum(str, Enum):
    rgb = "rgb"
    bgr = "bgr"
    yuv = "yuv"


class InputTensorEnum(str, Enum):
    nchw = "nchw"
    nhwc = "nhwc"


class ModelTypeEnum(str, Enum):
    embedding = "embedding"


class ModelConfig(frigate.detectors.detector_config.ModelConfig):
    model_type: ModelTypeEnum = Field(
        default=ModelTypeEnum.embedding, title="Object Detection Model Type"
    )

    database_path: str = Field(
        title="Database containing mapping from class names to feature vectors."
    )
    width_resize: int = Field(default=128, title="TODO.")
    height_resize: int = Field(default=128, title="TODO.")
    threshold_reid_neighbours: float = Field(title="TODO.")

    path: Optional[str] = Field(title="Custom Identifier model path.")
    labelmap_path: Optional[str] = Field(title="Label map for custom identifier.")
    width: int = Field(default=320, title="Identifier input width.")
    height: int = Field(default=320, title="Identifier input height.")


class BaseIdentifierConfig(BaseModel):
    # the type field must be defined in all subclasses
    type: str = Field(default="cpu", title="Identifier Type")
    model: ModelConfig = Field(
        default=None, title="Identifier specific model configuration."
    )

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = True
