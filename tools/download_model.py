"""Download a published model with its pinned size and SHA256; never overwrite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]


def verified_bytes(data, entry):
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('Downloaded model size or SHA256 does not match the release manifest')
    return data


def download(entry, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    request = urllib.request.Request(entry['url'], headers={'User-Agent': 'mlx90640-model-downloader'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = verified_bytes(response.read(entry['bytes'] + 1), entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f'.{output.name}.{uuid.uuid4().hex}.part')
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=['pt', 'onnx'], default='pt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/pretrained.json').read_text(encoding='utf-8'))
    entry = manifest['files'][args.format]
    path = download(entry, args.output)
    print(f'Verified {path.name}: SHA256 {entry["sha256"]}')


if __name__ == '__main__':
    main()
