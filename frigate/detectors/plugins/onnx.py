import logging

import numpy as np

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

        self.conf_th = 0.5  ##TODO: model config parameter
        self.nms_threshold = 0.4
        # self.trt_logger = TrtLogger()

        try:
            self.onnxruntime_session = onnxruntime.InferenceSession(
                detector_config.model.path,
                providers=[
                    detector_config.device,
                ],
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

    # TODO Directly copied, merge back to a single class
    def _postprocess_yolo(self, outputs, conf_th):
        """Postprocess TensorRT outputs.
        # Args
            trt_outputs: a list of 2 or 3 tensors, where each tensor
                        contains a multiple of 7 float32 numbers in
                        the order of [x, y, w, h, box_confidence, class_id, class_prob]
            conf_th: confidence threshold
        # Returns
            boxes, scores, classes
        """
        assert len(outputs) == 1
        predictions = outputs[0]

        predictions.shape[1] - 4

        detections = []
        for x in predictions:
            j = np.argmax(x[4:, :], axis=0)
            conf = np.take_along_axis(
                x[4:, :], np.expand_dims(j, axis=0), axis=0
            ).squeeze()
            m = conf > conf_th
            x = x[:, m]
            conf = conf[m]
            j = j[m]

            detections.append(np.hstack((x[:4].T, conf[..., None], j[..., None])))

        # filter low-conf detections and concatenate results of all yolo layers
        # detections = []
        # for o in trt_outputs:
        #     dets = o.reshape((-1, 7))
        #     dets = dets[dets[:, 4] * dets[:, 6] >= conf_th]
        #     detections.append(dets)
        # detections = np.concatenate(detections, axis=0)

        return detections

    def detect_raw(self, tensor_input):
        # Input tensor has the shape of the [height, width, 3]
        # Output tensor of float32 of shape [20, 6] where:
        # O - class id
        # 1 - score
        # 2..5 - a value between 0 and 1 of the box: [top, left, bottom, right]

        # normalize
        if self.input_dtype != np.uint8:
            tensor_input = tensor_input.astype(self.input_dtype)
            tensor_input /= 255.0

        outputs = self._do_inference(tensor_input)
        assert len(outputs) == 1

        raw_detections = self._postprocess_yolo(outputs, self.conf_th)

        if len(raw_detections) == 0:
            return np.zeros((20, 6), np.float32)

        assert len(raw_detections) == 1
        raw_detections = raw_detections[0]  # (xc, xy, w, h)

        h, w = tensor_input.shape[-2:]

        print(raw_detections[:, :4])
        # (xc, yc, w, h) [absolute] -> (x, y, w, h) [relative]
        raw_detections[:, 0], raw_detections[:, 2] = (
            raw_detections[:, 0] - raw_detections[:, 2] / 2
        ) / w, (raw_detections[:, 0] + raw_detections[:, 2] / 2) / w
        raw_detections[:, 1], raw_detections[:, 3] = (
            raw_detections[:, 1] - raw_detections[:, 3] / 2
        ) / h, (raw_detections[:, 1] + raw_detections[:, 3] / 2) / h

        print(raw_detections[:, :4])
        print("---")

        # raw_detections: Nx7 numpy arrays of
        #             [[x, y, w, h, box_confidence, class_id, class_prob],

        # Calculate score as box_confidence x class_prob
        # raw_detections[:, 4] = raw_detections[:, 4] * raw_detections[:, 6]
        # Reorder elements by the score, best on top, remove class_prob
        ordered = raw_detections[raw_detections[:, 4].argsort()[::-1]][:, 0:6]
        # transform width to right with clamp to 0..1
        # ordered[:, 2] = np.clip(ordered[:, 2] + ordered[:, 0], 0, 1)
        # transform height to bottom with clamp to 0..1
        # ordered[:, 3] = np.clip(ordered[:, 3] + ordered[:, 1], 0, 1)
        # put result into the correct order and limit to top 20
        detections = ordered[:, [5, 4, 1, 0, 3, 2]][:20]
        # pad to 20x6 shape
        append_cnt = 20 - len(detections)
        if append_cnt > 0:
            detections = np.append(
                detections, np.zeros((append_cnt, 6), np.float32), axis=0
            )

        return detections
