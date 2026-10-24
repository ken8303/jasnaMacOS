"""Opt-in benchmark variants; the shipping detector remains FP32 and uncompiled."""
from rfdetr_mlx_detector import RFDetrMLXDetector, mx


class ExperimentalRFDetrMLXDetector(RFDetrMLXDetector):
    def __init__(self, archive, *, max_select=64, precision='float32', compile_forward=False):
        super().__init__(archive, max_select=max_select)
        if precision not in ('float32', 'float16'):
            raise ValueError('precision must be float32 or float16')
        self.dtype = mx.float16 if precision == 'float16' else mx.float32
        self.weights = {key: value.astype(self.dtype) if value.dtype == mx.float32 else value
                        for key, value in self.weights.items()}
        mx.eval(self.weights)
        self.forward = mx.compile(self._forward) if compile_forward else self._forward

    def predict_raw(self, image):
        if image.ndim != 4 or image.shape[1:] != (768, 768, 3):
            raise ValueError('expected normalized RGB 768x768 image batch')
        result = self.forward(image.astype(self.dtype))
        mx.eval(result)
        return result
