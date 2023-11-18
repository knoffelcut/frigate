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
import pickle

import scipy.spatial.distance
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


pixel_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
pixel_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class_name_to_label = {
    'skapie': 0,
    'gertjie': 1,
    'lola': 2,
    'charlie': 3,
    'would-be-lee': 5
}
label_to_class_name = {v: k for k, v in class_name_to_label.items()}


def predict_reid(onnxruntime_session: onnxruntime.InferenceSession, image: np.ndarray):
    image = image[:, :, ::-1]  # BGR2RGB
    # image = image.astype(np.float32)/255.
    image = (image - pixel_mean)/pixel_std
    image = np.ascontiguousarray(image.transpose((2, 0, 1))[None, ...].astype(np.float32))

    onnxruntime_name_input = onnxruntime_session.get_inputs()[0].name
    onnxruntime_inputs = {onnxruntime_name_input: image}
    output = onnxruntime_session.run(None, onnxruntime_inputs)
    d = output[0][0]

    return d


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
                "/media/models/osnet_ain_x1_0_catcam_softmax_cosinelr_3.onnx",  # # TODO From config
                providers=providers
            )

            path_reid_db = "/media/models/feature_vectors_0d094e5a.pkl"  # TODO From config
            with open(path_reid_db, "rb") as f:
                self.feature_vectors_reid = pickle.load(f)

            self.X = []
            self.y = []
            for _, label, d in self.feature_vectors_reid:
                try:
                    label = class_name_to_label[label]
                except KeyError:
                    continue

                # print(label)
                # y.append(label)
                self.y.append(label)
                self.X.append(d)

            self.y = np.array(self.y)
            self.X = np.array(self.X)
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

            # nhwc
            height, width = tensor_input.shape[-3:-1]

            results_ = []
            for detection in results:
                # Crop and pad
                _, _, t, l, b, r = detection
                l = int(round(l*width))
                r = int(round(r*width))
                t = int(round(t*height))
                b = int(round(b*height))

                pl, l = -min(0, l), max(0, l)
                pt, t = -min(0, t), max(0, t)
                pr, r = max(0, r - width), min(r, width)
                pb, b = max(0, b - height), min(b, height)

                assert tensor_input.shape[0] == 1
                crop = tensor_input[0, t:b, l:r]
                if max((pl, pt, pr, pb)) > 0:
                    crop = np.pad(crop, ((pt, pb), (pl, pr), (0, 0)))
                crop = cv2.resize(crop, (256, 256))  # TODO Unhardcode

                d = predict_reid(self.onnxruntime_session_identification, crop)

                distances_cosine = scipy.spatial.distance.cdist(d[None, ...], self.X, metric='cosine')[0]
                distances = distances_cosine
                idx = np.where(distances < 0.15)[0]

                labels = self.y[idx]
                try:
                    mode = scipy.stats.mode(labels)
                    label = mode[0][0]
                    confidence = mode[1][0]/len(labels)
                except:
                    label = self.y[np.argmin(distances)]
                    confidence = 0.0

                if confidence > 0.5:
                    results_.append(
                        label,
                        confidence,
                        detection[2],
                        detection[3],
                        detection[4],
                        detection[5],
                    )

                # class_name = label_to_class_name[label]
                # print(class_name)

                results = results_
        return results
