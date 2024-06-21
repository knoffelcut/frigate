import logging
from abc import ABC, abstractmethod

import numpy as np

from frigate.detectors.detector_config import ModelTypeEnum

logger = logging.getLogger(__name__)


class DetectionApi(ABC):
    type_key: str

    @abstractmethod
    def __init__(self, detector_config):
        self.min_score = detector_config.model.min_score

    @abstractmethod
    def detect_raw(self, tensor_input):
        pass

    ## Takes in class ID, confidence score, and array of [x, y, w, h] that describes detection position,
    ## returns an array that's easily passable back to Frigate.
    def process_yolo(self, class_id, conf, pos):
        return [
            class_id,  # class ID
            conf,  # confidence score
            (pos[1] - (pos[3] / 2)) / self.h,  # y_min
            (pos[0] - (pos[2] / 2)) / self.w,  # x_min
            (pos[1] + (pos[3] / 2)) / self.h,  # y_max
            (pos[0] + (pos[2] / 2)) / self.w,  # x_max
        ]

    def process_results(self, results: np.ndarray):
        if self.model_type == ModelTypeEnum.ssd:
            detections = np.zeros((20, 6), np.float32)
            i = 0
            for object_detected in results.data[0, :]:
                if object_detected[0] != -1:
                    logger.debug(object_detected)
                if object_detected[2] < self.min_score or i == 20:
                    break
                detections[i] = [
                    object_detected[1],  # Label ID
                    float(object_detected[2]),  # Confidence
                    object_detected[4],  # y_min
                    object_detected[3],  # x_min
                    object_detected[6],  # y_max
                    object_detected[5],  # x_max
                ]
                i += 1
            return detections
        elif self.model_type == ModelTypeEnum.yolox:
            # [x, y, h, w, box_score, class_no_1, ..., class_no_80],
            results = results[None, ...]
            results[..., :2] = (results[..., :2] + self.grids) * self.expanded_strides
            results[..., 2:4] = np.exp(results[..., 2:4]) * self.expanded_strides
            image_pred = results[0, ...]

            class_conf = np.max(
                image_pred[:, 5 : 5 + self.num_classes], axis=1, keepdims=True
            )
            class_pred = np.argmax(image_pred[:, 5 : 5 + self.num_classes], axis=1)
            class_pred = np.expand_dims(class_pred, axis=1)

            # Below 0.3 is not eq. to min_score
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= 0.3).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            dets = np.concatenate((image_pred[:, :5], class_conf, class_pred), axis=1)
            dets = dets[conf_mask]

            ordered = dets[dets[:, 5].argsort()[::-1]][:20]

            detections = np.zeros((20, 6), np.float32)

            for i, object_detected in enumerate(ordered):
                detections[i] = self.process_yolo(
                    object_detected[6], object_detected[5], object_detected[:4]
                )
            return detections
        elif self.model_type == ModelTypeEnum.yolov8:
            output_data = np.transpose(results)
            scores = np.max(output_data[:, 4:], axis=1)
            if len(scores) == 0:
                return np.zeros((20, 6), np.float32)
            scores = np.expand_dims(scores, axis=1)
            # add scores to the last column
            dets = np.concatenate((output_data, scores), axis=1)
            # filter out lines with scores below threshold
            dets = dets[dets[:, -1] > self.min_score, :]
            # limit to top 20 scores, descending order
            ordered = dets[dets[:, -1].argsort()[::-1][:20]]
            detections = np.zeros((20, 6), np.float32)

            for i, object_detected in enumerate(ordered):
                detections[i] = self.process_yolo(
                    np.argmax(object_detected[4:-1]),
                    object_detected[-1],
                    object_detected[:4],
                )
            return detections
        elif self.model_type == ModelTypeEnum.yolov5:
            output_data = results
            # filter out lines with scores below threshold
            conf_mask = (output_data[:, 4] >= self.min_score).squeeze()
            output_data = output_data[conf_mask]
            # limit to top 20 scores, descending order
            ordered = output_data[output_data[:, 4].argsort()[::-1]][:20]

            detections = np.zeros((20, 6), np.float32)

            for i, object_detected in enumerate(ordered):
                detections[i] = self.process_yolo(
                    np.argmax(object_detected[5:]),
                    object_detected[4],
                    object_detected[:4],
                )
            return detections
