"""
Placeholder for tensorrt identifier
Primarily exists to bypass this issue: https://github.com/pydantic/pydantic/pull/3636
    i.e. more than one plugin must be declared otherwise config parsing fails
TODO Lost of function here can simply be imported from the corresponding detector
"""
import logging

import numpy as np

try:
    import tensorrt as trt
    from cuda import cuda

    TRT_SUPPORT = True
except ModuleNotFoundError:
    TRT_SUPPORT = False

from pydantic import Field
from typing_extensions import Literal

from frigate.identifiers.identifier_api import IdentificationApi
from frigate.identifiers.identifier_config import BaseIdentifierConfig

logger = logging.getLogger(__name__)

IDENTIFIER_KEY = "tensorrt"

if TRT_SUPPORT:

    class TrtLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, msg):
            logger.log(self.getSeverity(severity), msg)

        def getSeverity(self, sev: trt.ILogger.Severity) -> int:
            if sev == trt.ILogger.VERBOSE:
                return logging.DEBUG
            elif sev == trt.ILogger.INFO:
                return logging.INFO
            elif sev == trt.ILogger.WARNING:
                return logging.WARNING
            elif sev == trt.ILogger.ERROR:
                return logging.ERROR
            elif sev == trt.ILogger.INTERNAL_ERROR:
                return logging.CRITICAL
            else:
                return logging.DEBUG


# TODO Potentially move to other configs
# TODO Regardless, this is a copy of the corresponding in detectors/plugins/onnx.py
class TensorRTConfig(BaseIdentifierConfig):
    type: Literal[IDENTIFIER_KEY]
    device: int = Field(default=0, title="GPU Device Index")


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""

    def __init__(self, host_mem, device_mem, nbytes, size):
        self.host = host_mem
        err, self.host_dev = cuda.cuMemHostGetDevicePointer(self.host, 0)
        self.device = device_mem
        self.nbytes = nbytes
        self.size = size

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        cuda.cuMemFreeHost(self.host)
        cuda.cuMemFree(self.device)


# TODO Should move to configuration or get from configuration (likely there already)
pixel_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
pixel_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TensorRtDetector(IdentificationApi):
    type_key = IDENTIFIER_KEY

    def _do_inference(self, inputs):
        """do_inference"""
        raise NotImplementedError

    def __init__(self, identifier_config: TensorRTConfig):
        assert (
            TRT_SUPPORT
        ), f"TensorRT libraries not found, {IDENTIFIER_KEY} identifer not present"
        raise NotImplementedError

    def __del__(self):
        """Free CUDA memories."""
        if self.outputs is not None:
            del self.outputs
        if self.inputs is not None:
            del self.inputs
        if self.stream is not None:
            cuda.cuStreamDestroy(self.stream)
            del self.stream
        del self.engine
        del self.context
        del self.trt_logger
        cuda.cuCtxDestroy(self.cu_ctx)

    def detect_raw(self, tensor_input):
        raise NotImplementedError
