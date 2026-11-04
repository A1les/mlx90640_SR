"""Export a supplied checkpoint and verify float32 PyTorch/ONNX parity."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ..model import load_model
from ..inference import OnnxPredictor, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import onnx
    import onnxscript  # required by the explicitly selected exporter
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint)
    example = torch.full((1, 1, 24, 32), 30.)
    torch.onnx.export(model, (example,), str(args.output), input_names=['temperature_c'],
                      output_names=['sr_temperature_c'], opset_version=18, dynamo=True, external_data=False)
    onnx.checker.check_model(str(args.output), full_check=True)
    predictor = OnnxPredictor(args.output)
    rng = np.random.default_rng(20261101)
    cases = [np.full((24,32), t, np.float32) for t in [-10,25,35,60]]
    cases += [rng.uniform(20,45,(24,32)).astype(np.float32) for _ in range(12)]
    maximum = 0.
    with torch.inference_mode():
        for frame in cases:
            expected = model(torch.from_numpy(frame[None,None]))[0,0].numpy()
            maximum = max(maximum, float(np.max(np.abs(expected-predictor.predict(frame)))))
    if maximum > .001:
        raise AssertionError(f'ONNX parity error {maximum} C exceeds 0.001 C')
    result = dict(checkpoint_sha256=sha256(args.checkpoint), onnx_sha256=sha256(args.output),
                  selected_epoch=checkpoint['epoch'], maximum_absolute_difference_c=maximum,
                  scope='Numerical parity only; not physical accuracy or spatial resolution')
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
