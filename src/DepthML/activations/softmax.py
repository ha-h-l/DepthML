from typing import Any, Callable
from .base_activation import BaseActivation

from DepthTensor import Tensor, TensorData, Device, create_1in_1out

import DepthTensor as DTensor
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def op(
    x: TensorData, *, device: Device, axis: DTensor.Axis = -1, **kwds: Any
) -> Tensor:
    if device == "cpu":
        xp = np
    else:
        if cp is None:
            raise DTensor.CuPyNotFound(DTensor.CUPY_NOT_FOUND_MSG)
        xp = cp
    x_max = xp.max(x, axis=axis, keepdims=True)
    exp_x = xp.exp(x - x_max)
    return Tensor(exp_x / xp.sum(exp_x, axis=axis, keepdims=True), requires_grad=True)


def diff(
    result: Tensor,
    x: TensorData,
    device: DTensor.Device,
    axis: DTensor.Axis,
    **kwds: Any
) -> Callable[[], TensorData]:

    def dx() -> TensorData:
        if device == "cpu":
            xp = np
        else:
            if cp is None:
                raise DTensor.CuPyNotFound(DTensor.CUPY_NOT_FOUND_MSG)
            xp = cp
        prod = xp.sum(result.grad * result.data, axis=axis, keepdims=True)
        return result.data * (result.grad - prod)

    return dx


func = create_1in_1out(op, diff)


class Softmax(BaseActivation):
    def __init__(self, axis: DTensor.Axis = -1, name: str = "activation") -> None:
        super().__init__(name)
        self.axis = axis

    ###
    ### Block abstracts
    ###

    def call(self, X: Tensor, **kwargs: Any) -> Tensor:
        return func(X, axis=self.axis)
