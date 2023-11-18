"""
TODO Reid support was added the hardcoded, dodgy way:
    Just directly sticking it into this script
It should rather go through a whole separate idendification module, in `frigate/identification`,
    i.e. decoupled from the detector,
    with the corresponding entries in the config file,
    e.g. path to db
    detection classes to which re-id must be applied (per reid model)
"""

import logging

import numpy as np
import cv2

try:
    import onnxruntime

    ONNX_SUPPORT = True
except ModuleNotFoundError:
    ONNX_SUPPORT = False

from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

logger = logging.getLogger(__name__)

DETECTOR_KEY = "onnx"

onnx_type_to_numpy = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
}


class OnnxDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default="CPUExecutionProvider", title="Device Type")


class OnnxDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def _do_inference(self, inputs):
        """do_inference"""
        if isinstance(inputs, dict):
            onnxruntime_inputs = {
                k.name: inputs[k.name] for k in self.onnxruntime_session.get_inputs()
            }
        elif isinstance(inputs, np.ndarray):
            assert len(self.onnxruntime_session.get_inputs()) == 1
            onnxruntime_inputs = {self.onnxruntime_session.get_inputs()[0].name: inputs}
        else:
            onnxruntime_inputs = {
                self.onnxruntime_session.get_inputs()[i].name: input_tensor
                for i, input_tensor in enumerate(inputs)
            }
        return self.onnxruntime_session.run(None, onnxruntime_inputs)

    def __init__(self, detector_config: OnnxDetectorConfig):
        assert (
            ONNX_SUPPORT
        ), f"ONNX libraries not found, {DETECTOR_KEY} detector not present"

        self.model_type = detector_config.model.model_type

        self.h = detector_config.model.height
        self.w = detector_config.model.width

        self.nms_threshold = 0.3  # TODO As configurable parameter

        try:
            logger.debug(
                f"Loading ONNX Model ({detector_config.model.path}) to {detector_config.device,}"
            )
            providers = [detector_config.device, ]
            if 'CPUExecutionProvider' not in providers:
                providers += ['CPUExecutionProvider', ]
            self.onnxruntime_session = onnxruntime.InferenceSession(
                detector_config.model.path,
                providers=providers
            )

            # TODO MOVE
            self.onnxruntime_session_identification = onnxruntime.InferenceSession(
                "/media/models/osnet_ain_x1_0_catcam_softmax_cosinelr_3.onnx",  # TODO Not hardcoded
                providers=providers
            )
        except Exception as e:
            logger.error(e)
            raise RuntimeError("failed to create ONNX Runtime Session") from e

        input_tensor_description = self.onnxruntime_session.get_inputs()[0]
        self.input_shape = input_tensor_description.shape
        self.input_dtype = onnx_type_to_numpy[input_tensor_description.type]

        logger.debug(
            f"ONNX model loaded. Input shape is {self.input_shape} ({self.input_dtype})"
        )
        logger.debug(
            f"ONNX model version is {onnxruntime.__version__}",
        )

    def __del__(self):
        """Free CUDA memories."""
        del self.onnxruntime_session  # FWIW

    def detect_raw(self, tensor_input):
        # normalize
        if self.input_dtype != np.uint8:
            tensor_input = tensor_input.astype(self.input_dtype)
            tensor_input /= 255.0

        outputs = self._do_inference(tensor_input)
        assert len(outputs) == 1  # Single output tensor
        assert len(outputs[0]) == 1  # Batch Size == 1
        results = outputs[0][0]

        results = self.process_results(results)

        # for object_detected in results.data[0, :]:
        #     if object_detected[0] != -1:
        #         logger.debug(object_detected)
        #     if object_detected[2] < 0.1 or i == 20:
        #         break
        #     detections[i] = [
        #         object_detected[1],  # Label ID
        #         float(object_detected[2]),  # Confidence
        #         object_detected[4],  # y_min
        #         object_detected[3],  # x_min
        #         object_detected[6],  # y_max
        #         object_detected[5],  # x_max
        #     ]
        #     i += 1
        # return detections
        if len(results):
            idx = cv2.dnn.NMSBoxes(
                [[l, t, r - l, b - t] for _, _, t, l, b, r in results],
                [c for _, c, _, _, _, _ in results],
                0.0,  # Enforced by top-k below
                self.nms_threshold,
                top_k=20,
            )
            results = [results[i] for i in idx]

        return results
