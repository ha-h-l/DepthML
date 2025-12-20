from typing import Any, Callable
from .base_activation import BaseActivation

from DepthTensor import Tensor, TensorData, Device, create_1in_1out

import DepthTensor as DTensor


def op(x: TensorData, *, device: Device, alpha: float = 0.01, **kwds: Any) -> Tensor:
    return DTensor.maximum(alpha * x, x, requires_grad=True)


def diff(
    result: Tensor, x: TensorData, alpha: float = 0.01, **kwds: Any
) -> Callable[[], TensorData]:

    def dx() -> TensorData:
        return result.grad * DTensor.where(x > 0.0, 1.0, alpha).data

    return dx


func = create_1in_1out(op, diff)


class LeakyReLU(BaseActivation):
    def __init__(self, alpha: float = 0.01, name: str = "activation") -> None:
        super().__init__(name)
        self.alpha = alpha

    ###
    ### Block abstracts
    ###

    def call(self, X: Tensor, **kwargs: Any) -> Tensor:
        return func(X, alpha=self.alpha)
