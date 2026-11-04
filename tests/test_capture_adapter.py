import types
import unittest
import numpy as np
from mlx90640_pipeline.capture import make_sensor_class


class FakeDriver:
    class MLX90640:
        def _GetTa(self, frame): return 25.
        def _GetVdd(self, frame): return 3.3
        def _CalculateTo(self, frame, emissivity, tr, result):
            self.calls.append((emissivity,tr,frame[833]))
            parity=np.indices((24,32)).sum(axis=0).ravel()%2
            for i in np.flatnonzero(parity==frame[833]):
                result[i]=float(frame[i])
        def getFrame(self,result):
            self.calls=[]
            for page in (0,1):
                frame=list(range(832))+[0x1a81,page]
                self._CalculateTo(frame,.95,17.,result)


class CaptureTests(unittest.TestCase):
    def test_adapter_preserves_parent_conversion_and_records_both_subpages(self):
        sensor=make_sensor_class(FakeDriver)()
        result=sensor.read_native()
        np.testing.assert_array_equal(result,np.arange(768).reshape(24,32))
        self.assertEqual(result.dtype,np.float64)
        self.assertEqual(sensor.calls,[(.95,17.,0),(.95,17.,1)])
        self.assertEqual(np.asarray(sensor.raw_subpages).shape,(2,834))
        self.assertEqual([x['subpage'] for x in sensor.conversion_inputs],[0,1])


if __name__ == '__main__':
    unittest.main()
