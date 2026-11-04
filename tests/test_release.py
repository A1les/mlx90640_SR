"""Integrity boundaries for the independently downloadable model artifacts."""
import hashlib
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('model_download', ROOT / 'tools/download_model.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class ReleaseIntegrityTests(unittest.TestCase):
    def test_rejects_corruption_with_unchanged_length(self):
        entry = {'bytes': 4, 'sha256': hashlib.sha256(b'model'[:4]).hexdigest()}
        with self.assertRaises(ValueError):
            download.verified_bytes(b'fail', entry)

    def test_rejects_truncation(self):
        entry = {'bytes': 5, 'sha256': hashlib.sha256(b'model').hexdigest()}
        with self.assertRaises(ValueError):
            download.verified_bytes(b'mode', entry)

    def test_verified_payload_retains_exact_bytes(self):
        payload = b'\x00model\xff'
        entry = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        self.assertEqual(download.verified_bytes(payload, entry), payload)


if __name__ == '__main__':
    unittest.main()
