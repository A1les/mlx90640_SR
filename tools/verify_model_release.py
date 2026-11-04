"""Verify published artifacts and CPU PyTorch/ONNX numerical agreement."""
import json
from pathlib import Path
import tempfile
import numpy as np
from download_model import ROOT, download
from mlx90640_pipeline.inference import TorchPredictor, OnnxPredictor


def main():
    manifest = json.loads((ROOT / 'configs/pretrained.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as directory:
        paths = {}
        for kind, entry in manifest['files'].items():
            paths[kind] = download(entry, Path(directory) / entry['name'])
        pt = TorchPredictor(paths['pt'])
        onnx = OnnxPredictor(paths['onnx'])
        rng = np.random.default_rng(20261104)
        cases = [np.full((24, 32), t, np.float32) for t in (0, 20, 30, 60)]
        cases += [rng.uniform(20, 45, (24, 32)).astype(np.float32) for _ in range(12)]
        error = max(float(np.max(np.abs(pt.predict(x) - onnx.predict(x)))) for x in cases)
        if error >= .001:
            raise AssertionError(f'PyTorch/ONNX maximum difference {error} C exceeds 0.001 C')
        print(json.dumps({'inputs': len(cases), 'maximum_absolute_difference_c': error,
                          'model': manifest['model_id'], 'passed': True}))


if __name__ == '__main__':
    main()
