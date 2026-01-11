from typing import Literal, TypeAlias
from DepthTensor.typing import Shape, Device
from .base_initializer import BaseInitializer
from DepthTensor import Tensor, random, DTypeLike, float32
import numpy as np

FanMode: TypeAlias = Literal["fan_in", "fan_out"]


class KaimingUniform(BaseInitializer):
    def __init__(
        self,
        a: float = 0,
        mode: FanMode = "fan_in",
        name: str = "kaiming_uniform",
    ) -> None:
        super().__init__(name)
        self.a = a
        self.mode = mode

    def __call__(
        self,
        shape: Shape,
        device: Device,
        requires_grad: bool,
        dtype: DTypeLike = float32,
    ) -> Tensor:
        fan_in, fan_out = self.compute_fans(shape)
        fan = fan_in if self.mode == "fan_in" else fan_out
        gain = np.sqrt(2.0 / (1 + self.a**2))
        std = gain / np.sqrt(fan)
        limit = np.sqrt(3.0) * std

        return random.uniform(
            low=-limit,
            high=limit,
            size=shape,
            device=device,
            requires_grad=requires_grad,
            dtype=dtype,
        )


class KaimingNormal(BaseInitializer):
    def __init__(
        self,
        a: float = 0,
        mode: FanMode = "fan_in",
        name: str = "kaiming_normal",
    ) -> None:
        super().__init__(name)
        self.a = a
        self.mode = mode

    def __call__(
        self,
        shape: Shape,
        device: Device,
        requires_grad: bool,
        dtype: DTypeLike = float32,
    ) -> Tensor:
        fan_in, fan_out = self.compute_fans(shape)
        fan = fan_in if self.mode == "fan_in" else fan_out
        gain = np.sqrt(2.0 / (1 + self.a**2), dtype=dtype)
        std = gain / np.sqrt(fan, dtype=dtype)
        return (
            random.randn(
                *shape, device=device, requires_grad=requires_grad, dtype=dtype
            )
            * std
        )
