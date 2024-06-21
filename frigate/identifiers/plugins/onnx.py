import logging
import pickle

import faiss
import numpy as np
import scipy.spatial.distance

try:
    import onnxruntime

    ONNX_SUPPORT = True
except ModuleNotFoundError:
    ONNX_SUPPORT = False

from pydantic import Field
from typing_extensions import Literal

from frigate.identifiers.identifier_api import IdentificationApi
from frigate.identifiers.identifier_config import BaseIdentifierConfig

logger = logging.getLogger(__name__)

IDENTIFIER_KEY = "onnx"

onnx_type_to_numpy = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
}


# TODO Potentially move to other configs
# TODO Regardless, this is a copy of the corresponding in detectors/plugins/onnx.py
class OnnxIdentifierConfig(BaseIdentifierConfig):
    type: Literal[IDENTIFIER_KEY]
    device: str = Field(default="CPUExecutionProvider", title="Device Type")


# TODO Should move to configuration or get from configuration (likely there already)
pixel_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
pixel_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OnnxDetector(IdentificationApi):
    type_key = IDENTIFIER_KEY

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

    def __init__(self, identifier_config: OnnxIdentifierConfig):
        assert (
            ONNX_SUPPORT
        ), f"ONNX libraries not found, {IDENTIFIER_KEY} identifier not present"

        self.model_type = identifier_config.model.model_type

        self.nms_threshold = 0.3  # TODO As configurable parameter

        self.h_resize = identifier_config.model.height_resize
        self.w_resize = identifier_config.model.width_resize
        self.h = identifier_config.model.height
        self.w = identifier_config.model.width
        self.threshold_reid_neighbours = (
            identifier_config.model.threshold_reid_neighbours
        )

        self.il, self.it, self.ir, self.ib = None, None, None, None
        if (self.h_resize, self.w_resize) != (
            self.h,
            self.w,
        ):
            self.il = (self.w_resize - self.w) // 2
            self.it = (self.h_resize - self.h) // 2
            self.ir = self.il + self.w
            self.ib = self.it + self.h

        try:
            logger.debug(
                f"Loading ONNX Model ({identifier_config.model.path}) to {identifier_config.device,}"
            )
            providers = [
                identifier_config.device,
            ]
            if "CPUExecutionProvider" not in providers:
                providers += [
                    "CPUExecutionProvider",
                ]

            self.onnxruntime_session = onnxruntime.InferenceSession(
                # detector_config.model.path_reidentification,
                identifier_config.model.path,
                providers=providers,
            )
            # TODO Run check to assert preferred/specified providers are included

            with open(
                identifier_config.model.database_path,
                "rb",
            ) as f:
                self.feature_vectors_reid = pickle.load(f)

            self.X = []
            self.y = []
            # Must reverse this way since merged_labelmap is prefilled till 91
            class_name_to_label = {
                identifier_config.model.merged_labelmap[k]: k
                for k in sorted(
                    [k for k in identifier_config.model.merged_labelmap.keys()]
                )[::-1]
            }
            for _, label, d in self.feature_vectors_reid:
                label = class_name_to_label[label]

                # print(label)
                # y.append(label)
                self.y.append(label)
                self.X.append(d)

            self.y_unknown = class_name_to_label["unknown"]
            self.y = np.array(self.y)
            self.X = np.array(self.X)

            faiss.normalize_L2(self.X)
            self.index = faiss.IndexFlatIP(self.X.shape[1])
            self.index.add(self.X)
        except Exception as e:
            logger.exception(e)
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
        try:
            del self.onnxruntime_session  # FWIW
        except AttributeError:
            pass

    def detect_raw(self, tensor_input):
        # normalize
        if self.input_dtype != np.uint8:
            tensor_input = tensor_input.astype(self.input_dtype)
            tensor_input /= 255.0
        tensor_input = (tensor_input - pixel_mean[..., None, None]) / pixel_std[
            ..., None, None
        ]

        outputs = self._do_inference(tensor_input)
        assert len(outputs) == 1  # Single output tensor
        assert len(outputs[0]) == 1  # Batch Size == 1
        embedding = outputs[0]

        faiss.normalize_L2(embedding)
        idx = self.index.range_search(embedding, 1 - self.threshold_reid_neighbours)[2]

        labels = self.y[idx]

        if len(labels) == 0:
            confidence = 1.0
            label = self.y_unknown
        else:
            mode = scipy.stats.mode(labels, keepdims=False)
            label = mode[0]
            confidence = mode[1] / len(labels)

        results = np.zeros((20, 6), dtype=np.float32)
        results[0, 0] = label
        results[0, 1] = confidence

        return results
