import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class IdentificationApi(ABC):
    type_key: str

    @abstractmethod
    def __init__(self, identifier_config):
        pass

    @abstractmethod
    def detect_raw(self, frame, detections):
        pass
