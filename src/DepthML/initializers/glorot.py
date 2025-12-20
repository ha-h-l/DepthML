from DepthTensor.typing import Shape, Device
from .base_initializer import BaseInitializer

from DepthTensor import Tensor, random

import numpy as np


class GlorotUniform(BaseInitializer):
    def __init__(self, name: str = "glorot_uniform") -> None:
        super().__init__(name)

    def __call__(self, shape: Shape, device: Device, requires_grad: bool) -> Tensor:
        fan_in, fan_out = self.compute_fans(shape)
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return random.uniform(
            low=-limit,
            high=limit,
            size=shape,
            device=device,
            requires_grad=requires_grad,
        )


class GlorotNormal(BaseInitializer):
    def __init__(self, name: str = "glorot_normal") -> None:
        super().__init__(name)

    def __call__(self, shape: Shape, device: Device, requires_grad: bool) -> Tensor:
        fan_in, fan_out = self.compute_fans(shape)
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return random.randn(*shape, device=device, requires_grad=requires_grad) * std
