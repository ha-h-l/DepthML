from abc import abstractmethod
from ..typing import tensor_types

from DepthTensor import Tensor


class BaseInitializer:
    def __init__(self, name: str = "initializer") -> None:
        self.name = name

    @abstractmethod
    def __call__(
        self,
        shape: tensor_types.Shape,
        device: tensor_types.Device,
        requires_grad: bool,
    ) -> Tensor:
        raise NotImplementedError

    def compute_fans(self, shape: tensor_types.Shape) -> tuple[float, float]:
        if len(shape) == 2:
            # linear Layer: (in, out)
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4:
            # conv Layer: (out_channels, in_channels, kH, kW)
            # fan_in = in_channels * kH * kW
            # fan_out = out_channels * kH * kW
            size = shape[2] * shape[3]
            fan_in = shape[1] * size
            fan_out = shape[0] * size
        else:
            # fallback for 1D or other shapes
            fan_in = shape[0]
            fan_out = shape[0]

        return fan_in, fan_out
