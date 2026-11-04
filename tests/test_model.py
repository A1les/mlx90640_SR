import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch optional dependency not installed')
class ModelTests(unittest.TestCase):
    def test_initialization_and_parameter_count(self):
        import torch
        from torch.nn import functional as F
        from mlx90640_pipeline.model import ThermalSRx2
        model=ThermalSRx2()
        self.assertEqual(sum(p.numel() for p in model.parameters()),16732)
        x=torch.randn(2,1,24,32)+30
        np.testing.assert_array_equal(model(x).detach().numpy(),F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False).numpy())

    def test_shift_equivariance_and_finite_training_gradients(self):
        import torch
        from mlx90640_pipeline.model import ThermalSRx2
        from mlx90640_pipeline.training.experiment import degradation, edge_mae_per_frame
        torch.manual_seed(1)
        model=ThermalSRx2()
        # Nonzero residual weights ensure the shift test covers both branches.
        torch.nn.init.normal_(model.head.weight,std=.01)
        target=torch.randn(2,1,48,64)+30
        x,_=degradation(target,.5,.1,torch.Generator().manual_seed(1))
        y=model(x)
        self.assertEqual(tuple(y.shape),(2,1,48,64))
        error=(model(x+10)-(y+10)).abs().max().item()
        self.assertLess(error,1e-4)
        loss=(y-target).abs().mean()+.2*edge_mae_per_frame(y,target).mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
