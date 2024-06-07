import logging
import pickle

import cv2
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


def predict_reid(onnxruntime_session: onnxruntime.InferenceSession, image: np.ndarray):
    image = image[:, :, ::-1]  # BGR2RGB
    # image = image.astype(np.float32)/255.
    image = (image - pixel_mean) / pixel_std
    image = np.ascontiguousarray(
        image.transpose((2, 0, 1))[None, ...].astype(np.float32)
    )

    onnxruntime_name_input = onnxruntime_session.get_inputs()[0].name
    onnxruntime_inputs = {onnxruntime_name_input: image}
    output = onnxruntime_session.run(None, onnxruntime_inputs)
    d = output[0][0]

    return d


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
                try:
                    label = class_name_to_label[label]
                except KeyError:
                    continue

                # print(label)
                # y.append(label)
                self.y.append(label)
                self.X.append(d)

            self.y_unknown = class_name_to_label["unknown"]
            self.y = np.array(self.y)
            self.X = np.array(self.X)
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
        del self.onnxruntime_session  # FWIW

    def detect_raw(self, tensor_input):
        raise NotImplementedError  # Process frame + detections from here

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
                0.5,  # Enforced by top-k below
                self.nms_threshold,
                top_k=20,
            )
            results = [results[i] for i in idx]

            # nchw
            height, width = tensor_input.shape[-2:]

            k = 0
            results_ = np.zeros((20, 6), dtype=np.float32)
            for detection in results:
                # Crop and pad
                _, _, t, l, b, r = detection
                l = int(round(l * width))
                r = int(round(r * width))
                t = int(round(t * height))
                b = int(round(b * height))

                pl, l = -min(0, l), max(0, l)
                pt, t = -min(0, t), max(0, t)
                pr, r = max(0, r - width), min(r, width)
                pb, b = max(0, b - height), min(b, height)

                assert tensor_input.shape[0] == 1
                crop = tensor_input[0, :, t:b, l:r]
                crop = crop.transpose((1, 2, 0))
                if max((pl, pt, pr, pb)) > 0:
                    crop = np.pad(crop, ((pt, pb), (pl, pr), (0, 0)))
                crop = cv2.resize(
                    crop, (self.w_identification_resize, self.h_identification_resize)
                )
                if self.il is not None:
                    crop = crop[self.it : self.ib, self.il : self.ir]

                d = predict_reid(self.onnxruntime_session_identification, crop)

                distances_cosine = scipy.spatial.distance.cdist(
                    d[None, ...], self.X, metric="cosine"
                )[0]
                distances = distances_cosine
                idx = np.where(distances < self.threshold_reid_neighbours)[0]

                labels = self.y[idx]

                if len(labels) == 0:
                    confidence = 1.0
                    label = self.y_unknown
                else:
                    mode = scipy.stats.mode(labels, keepdims=False)
                    label = mode[0]
                    confidence = mode[1] / len(labels)

                if confidence > 0.5:
                    results_[k] = (
                        label,
                        confidence,
                        detection[2],
                        detection[3],
                        detection[4],
                        detection[5],
                    )
                    k += 1

            results = np.array(results_).reshape((len(results_), 6))
        return results
