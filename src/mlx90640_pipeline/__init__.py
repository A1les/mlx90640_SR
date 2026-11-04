"""MLX90640 temperature processing and reconstruction.

``pipeline.ThermalPipeline`` combines motion-adaptive deinterlacing with both
CNN/EMA processing orders. The deinterlaced output preserves current-subpage
and unselected observations; subsequent temporal/CNN arrays are estimates.
``inference.TorchPredictor`` and ``inference.OnnxPredictor`` load explicit model
paths. Published model identities and downloads are in configs/pretrained.json.
No dataset, model or device configuration is downloaded implicitly.
"""
__version__ = "0.1.0"
