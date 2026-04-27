import logging

import numpy as np
import openvino.runtime as ov
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig, ModelTypeEnum

logger = logging.getLogger(__name__)

DETECTOR_KEY = "openvino"


class OvDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default=None, title="Device Type")


class OvDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: OvDetectorConfig):
        super().__init__(detector_config)
        self.ov_core = ov.Core()
        self.ov_model = self.ov_core.read_model(detector_config.model.path)
        self.ov_model_type = detector_config.model.model_type

        self.h = detector_config.model.height
        self.w = detector_config.model.width

        self.interpreter = self.ov_core.compile_model(
            model=self.ov_model, device_name=detector_config.device
        )

        logger.info(f"Model Input Shape: {self.interpreter.input(0).shape}")
        self.output_indexes = 0

        while True:
            try:
                tensor_shape = self.interpreter.output(self.output_indexes).shape
                logger.info(f"Model Output-{self.output_indexes} Shape: {tensor_shape}")
                self.output_indexes += 1
            except Exception:
                logger.info(f"Model has {self.output_indexes} Output Tensors")
                break
        if self.ov_model_type == ModelTypeEnum.yolox:
            self.num_classes = tensor_shape[2] - 5
            logger.info(f"YOLOX model has {self.num_classes} classes")
            self.set_strides_grids()

    def set_strides_grids(self):
        grids = []
        expanded_strides = []

        strides = [8, 16, 32]

        hsizes = [self.h // stride for stride in strides]
        wsizes = [self.w // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        self.grids = np.concatenate(grids, 1)
        self.expanded_strides = np.concatenate(expanded_strides, 1)

    def detect_raw(self, tensor_input):
        infer_request = self.interpreter.create_infer_request()
        infer_request.infer([tensor_input])

        results = infer_request.get_output_tensor()
        assert len(results[0]) == 1  # Batch Size == 1
        results = results[0][0]

        return self.process_results(results)
