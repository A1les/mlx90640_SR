"""Explicit, interchangeable inference backends; no dependency installation."""
from pathlib import Path
import hashlib
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_array(frame):
    a = np.asarray(frame, np.float32)
    if a.shape != (24, 32) or not np.isfinite(a).all():
        raise ValueError('Expected finite 24x32 temperatures in degrees Celsius')
    return a


def sr_array(frame):
    a = np.asarray(frame, np.float32)
    if a.shape != (48, 64) or not np.isfinite(a).all():
        raise ValueError('Model must return finite 48x64 estimated temperatures')
    return a


class BilinearBaseline:
    """Runnable baseline for the synthetic demo, explicitly not a trained CNN."""
    kind = 'bilinear_baseline_not_cnn'
    model_sha256 = None

    def predict(self, frame):
        import cv2
        return cv2.resize(native_array(frame), (64, 48), interpolation=cv2.INTER_LINEAR)


class TorchPredictor:
    kind = 'thermal_sr_x2_pytorch'

    def __init__(self, checkpoint, device='cpu', expected_sha256=None):
        import torch
        from .model import load_model
        self.model_sha256 = sha256(checkpoint)
        if expected_sha256 and self.model_sha256 != expected_sha256:
            raise ValueError('Checkpoint SHA256 does not match the expected model')
        self.device = device
        self.model, self.checkpoint = load_model(checkpoint, device)
        torch.set_num_threads(2)

    def predict(self, frame):
        import torch
        a = native_array(frame)
        with torch.inference_mode():
            out = self.model(torch.from_numpy(a[None, None]).to(self.device))
        return sr_array(out[0, 0].cpu().numpy())


class OnnxPredictor:
    kind = 'thermal_sr_x2_onnx'

    def __init__(self, model, threads=2, expected_sha256=None):
        import onnxruntime as ort
        self.model_sha256 = sha256(model)
        if expected_sha256 and self.model_sha256 != expected_sha256:
            raise ValueError('ONNX SHA256 does not match the expected model')
        if threads < 1:
            raise ValueError('threads must be positive')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].shape != [1, 1, 24, 32]
                or inputs[0].type != 'tensor(float)' or len(outputs) != 1
                or outputs[0].shape != [1, 1, 48, 64]):
            raise ValueError('Expected a fixed-shape float32 Celsius-domain x2 model')
        self.input_name = inputs[0].name

    def predict(self, frame):
        out = self.session.run(None, {self.input_name: native_array(frame)[None, None]})[0]
        return sr_array(out[0, 0])
