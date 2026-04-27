import functools
import logging
from abc import ABC, abstractmethod

import numpy as np
import scipy.special

from frigate.detectors.detector_config import ModelTypeEnum

logger = logging.getLogger(__name__)


@functools.lru_cache
def nanodet_center_priors(
    input_height: int, input_width: int, strides: tuple, dtype: type
):
    def get_single_level_center_priors(featmap_size, stride, dtype):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (np.arange(w, dtype=dtype)) * stride
        y_range = (np.arange(h, dtype=dtype)) * stride
        y, x = np.meshgrid(y_range, x_range, indexing="ij")
        y = y.flatten()
        x = x.flatten()
        strides = np.full(x.shape[0], stride)
        priors = np.stack([x, y, strides, strides], axis=-1)
        return priors

    featmap_sizes = [
        (
            int(np.ceil(input_height / stride)),
            int(np.ceil(input_width) / stride),
        )
        for stride in strides
    ]
    mlvl_center_priors = [
        get_single_level_center_priors(
            featmap_sizes[i],
            stride,
            dtype,
        )
        for i, stride in enumerate(strides)
    ]
    center_priors = np.concatenate(mlvl_center_priors, axis=0)

    return center_priors


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
        elif self.model_type == ModelTypeEnum.nanodet_plus:

            def distance2bbox(points, distance, max_shape=None):
                """Decode distance prediction to bounding box.

                Args:
                    points (Tensor): Shape (n, 2), [x, y].
                    distance (Tensor): Distance from the given point to 4
                        boundaries (left, top, right, bottom).
                    max_shape (tuple): Shape of the image.

                Returns:
                    Tensor: Decoded bboxes.
                """
                x1 = points[..., 0] - distance[..., 0]
                y1 = points[..., 1] - distance[..., 1]
                x2 = points[..., 0] + distance[..., 2]
                y2 = points[..., 1] + distance[..., 3]
                if max_shape is not None:
                    x1 = np.clip(x1, 0, max_shape[1])
                    y1 = np.clip(y1, 0, max_shape[0])
                    x2 = np.clip(x2, 0, max_shape[1])
                    y2 = np.clip(y2, 0, max_shape[0])
                return np.stack([x1, y1, x2, y2], -1)

            # TODO From parameters
            reg_max = 7
            strides = (8, 16, 32, 64)

            num_classes = results.shape[-1] - 4 * (reg_max + 1)
            cls_scores, bbox_preds = results[:, :num_classes], results[:, num_classes:]

            center_priors = nanodet_center_priors(
                self.h, self.w, strides, results[0].dtype
            )

            x = bbox_preds.reshape(bbox_preds.shape[0], 4, reg_max + 1)
            x = scipy.special.softmax(x, axis=-1)
            x = np.dot(x, np.linspace(0, reg_max, reg_max + 1))

            dis_preds = x * center_priors[..., 2, None]
            bboxes = distance2bbox(
                center_priors[..., :2], dis_preds, max_shape=(self.h, self.w)
            )

            class_ids = np.argmax(cls_scores, axis=1)
            scores = np.max(cls_scores, axis=1)

            detections = np.zeros((20, 6), dtype=np.float32)
            for i, j in enumerate(np.argsort(scores)[::-1][:20]):
                detections[i, 0] = class_ids[j]
                detections[i, 1] = scores[j]
                detections[i, 2] = bboxes[j, 1] / self.h
                detections[i, 3] = bboxes[j, 0] / self.w
                detections[i, 4] = bboxes[j, 3] / self.h
                detections[i, 5] = bboxes[j, 2] / self.w
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
            dets = dets[dets[:, -1] >= self.min_score, :]
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
        elif self.model_type == ModelTypeEnum.yolov10:
            # filter out lines with scores below threshold
            dets = results[results[:, 4] >= self.min_score, :]
            # limit to top 20 scores, descending order
            ordered = dets[dets[:, 4].argsort()[::-1][:20]]
            detections = np.zeros((20, 6), np.float32)

            for i, object_detected in enumerate(ordered):
                detections[i] = [
                    object_detected[5],
                    object_detected[4],
                    object_detected[1] / self.h,
                    object_detected[0] / self.w,
                    object_detected[3] / self.h,
                    object_detected[2] / self.w,
                ]
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

        raise NotImplementedError
