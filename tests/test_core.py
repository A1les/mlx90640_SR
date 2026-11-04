import copy
import json
from pathlib import Path
import tempfile
import os
import uuid
import shutil
import unittest
import numpy as np
from mlx90640_pipeline.calibration import fit_profile, save_profile
from mlx90640_pipeline.core.frequency import filter_temperature
from mlx90640_pipeline.core.motion import load_profile
from mlx90640_pipeline.core.postprocess import PostprocessPipeline
from mlx90640_pipeline.core.spatial import PARITY
from mlx90640_pipeline.demo import synthetic_recording
from mlx90640_pipeline.inference import BilinearBaseline
from mlx90640_pipeline.pipeline import ThermalPipeline
from mlx90640_pipeline.recording import load_recording, write_npz
from mlx90640_pipeline.viewer import render


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.static = synthetic_recording(180, moving=False)
        cls.profile = fit_profile(cls.static)

    def test_latest_and_unselected_observations_are_byte_identical(self):
        r = synthetic_recording(150)
        pipe = ThermalPipeline(self.profile, r.factory, BilinearBaseline())
        values, flags = pipe.sequence(r)
        raw = r.arrays['raw_temperature_c']
        latest = PARITY[None] == r.arrays['latest_subpage'][:,None,None]
        self.assertGreater(int(values['repair_mask'].sum()), 0)
        self.assertFalse(np.any(values['repair_mask'] & latest))
        keep = ~values['repair_mask']
        self.assertEqual(values['deinterlaced_c'][keep].tobytes(), raw[keep].tobytes())
        self.assertEqual(values['deinterlaced_c'].dtype, raw.dtype)
        pipe.postprocess.check_saved(values['deinterlaced_c'], values, flags)

    def test_noise_profile_is_specific_to_sensor(self):
        wrong = dict(self.static.factory, eeprom_sha256='1'*64)
        with self.assertRaises(ValueError):
            ThermalPipeline(self.profile, wrong, BilinearBaseline())

    def test_invalid_pair_and_gap_preserve_raw_and_reset_history(self):
        r = synthetic_recording(80)
        p = ThermalPipeline(self.profile, r.factory, BilinearBaseline())
        raw = r.arrays['raw_temperature_c'][60]
        p.update(raw, 1., 0, 0x1a81)
        for t, pair_ok in [(1.125, False), (2., True), (1.5, True)]:
            out, state = p.update(raw, t, 0, 0x1a81, pair_ok)
            self.assertTrue(state['repair_history_reset'])
            self.assertTrue(state['temporal_history_reset'])
            self.assertEqual(out['deinterlaced_c'].tobytes(), raw.tobytes())
            self.assertTrue(np.all(out['temporal_alpha'] == 1))

    def test_static_average_reduces_independent_noise(self):
        p = PostprocessPipeline(BilinearBaseline(), compare_order=True)
        rng = np.random.default_rng(3)
        raw, filtered = [], []
        empty = np.zeros((24,32), bool)
        for i in range(250):
            f = (30 + rng.normal(0,.2,(24,32))).astype(np.float32)
            out, _ = p.update(f, empty, i/8, False)
            if i > 30:
                raw.append(f); filtered.append(out['native_temporal_c'])
        ratio = np.std(filtered,axis=0).mean()/np.std(raw,axis=0).mean()
        self.assertTrue(.52 < ratio < .64, ratio)

    def test_scene_step_bypasses_both_orders_and_clears_history(self):
        p = PostprocessPipeline(BilinearBaseline(), compare_order=True)
        empty = np.zeros((24,32), bool)
        p.update(np.full((24,32),26.), empty, 0., False)
        for i in range(1,5):
            out, state = p.update(np.full((24,32),34.), empty, i/8, False)
            self.assertTrue(state['temporal_passthrough'])
            np.testing.assert_array_equal(out['final_temperature_c'], out['pre_cnn_temperature_c'])
            np.testing.assert_array_equal(out['final_temperature_c'], 34.)

    def test_repaired_region_releases_both_parities_for_two_frames(self):
        p = PostprocessPipeline(BilinearBaseline(), compare_order=True)
        f = np.full((24,32),30.)
        empty = np.zeros((24,32), bool)
        region = empty.copy(); region[10,10] = True
        p.update(f,empty,0.,False)
        for i, mask in [(1,region),(2,empty)]:
            out, _ = p.update(f,mask,i/8,False)
            self.assertTrue(np.all(out['temporal_native_alpha'][9:12,9:12] == 1))
        out, _ = p.update(f,empty,3/8,False)
        self.assertTrue(np.all(out['temporal_native_alpha'] == .5))

    def test_frequency_filter_preserves_constant_field(self):
        for value in [0.,25.,-5.,60.]:
            out = filter_temperature(np.full((24,32),value), width=.22, strength=.75, boundary='reflect8')
            np.testing.assert_array_equal(out, value)

    def test_recording_roundtrip_and_no_overwrite(self):
        parent = Path(os.environ.get('MLX90640_TEST_TMPDIR', tempfile.gettempdir())).resolve()
        d = parent / ('mlx90640_test_' + uuid.uuid4().hex)
        d.mkdir()
        try:
            path=d/'recording.npz'
            write_npz(path,self.static.arrays,self.static.metadata,self.static.factory)
            r=load_recording(path)
            np.testing.assert_array_equal(r.arrays['raw_temperature_c'], self.static.arrays['raw_temperature_c'])
            self.assertTrue(r.arrays['pair_ok'].all())
            with self.assertRaises(FileExistsError):
                write_npz(path,self.static.arrays,self.static.metadata,self.static.factory)
            profile_path=d/'profile.npz'
            save_profile(profile_path,self.profile)
            p=load_profile(profile_path)
            np.testing.assert_array_equal(p[0],self.profile[0])
        finally:
            if d.resolve().parent != parent:
                raise AssertionError('Unexpected temporary directory location')
            shutil.rmtree(d)

    def test_invalid_subpages_are_rejected(self):
        r=copy.deepcopy(self.static)
        r.arrays['raw_subpages_u16'][0,:,833]=0
        with self.assertRaises(ValueError):
            r.validate()

    def test_render_does_not_change_temperatures(self):
        p=ThermalPipeline(self.profile,self.static.factory,BilinearBaseline())
        out,_=p.update(self.static.arrays['raw_temperature_c'][0],0.,0,0x1a81)
        before={k:v.tobytes() for k,v in out.items()}
        image=render(out,scale=2)
        self.assertEqual(image.shape,(192,384,3))
        self.assertEqual(before,{k:v.tobytes() for k,v in out.items()})


if __name__ == '__main__':
    unittest.main()
