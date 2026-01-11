from DepthTensor.typing import Shape, Device
from .base_initializer import BaseInitializer

from DepthTensor import Tensor, zeros, DTypeLike, float32


class Zeros(BaseInitializer):
    def __init__(self, name: str = "zeros") -> None:
        super().__init__(name)

    def __call__(
        self,
        shape: Shape,
        device: Device,
        requires_grad: bool,
        dtype: DTypeLike = float32,
    ) -> Tensor:
        return zeros(
            shape=shape, device=device, requires_grad=requires_grad, dtype=dtype
        )
