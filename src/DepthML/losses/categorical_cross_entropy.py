from typing import Any, Callable
import numpy as np

from .base_loss import BaseLoss
from DepthTensor import Tensor, TensorData, Device, create_2in_1out

EPSILON = 1e-7


def op(x1: TensorData, x2: TensorData, *, device: Device, **kwargs: Any) -> Tensor:
    y_pred_clipped = np.clip(x2, EPSILON, 1.0 - EPSILON)
    # L = - sum(y_true * log(y_pred)) / N
    return Tensor(-np.mean(x1 * np.log(y_pred_clipped)), device=device)


def diff(
    result: Tensor, x1: TensorData, x2: TensorData, **kwargs: Any
) -> tuple[Callable[[], TensorData], Callable[[], TensorData]]:

    y_pred_clipped = np.clip(x2, EPSILON, 1.0 - EPSILON)
    a = 1.0 / x1.size

    def dx1() -> TensorData:
        # dL/d(true) = -log(pred)
        return result.grad * a * -np.log(y_pred_clipped)  # type: ignore

    def dx2() -> TensorData:
        # dL/d(pred) = -(true / pred)
        return result.grad * a * -(x1 / y_pred_clipped)  # type: ignore

    return dx1, dx2


func = create_2in_1out(op, diff)


class CategoricalCrossentropy(BaseLoss):
    def __init__(self, name: str = "categorical_crossentropy") -> None:
        super().__init__(name)

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return super().__call__(y_true, y_pred)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return func(y_true, y_pred)
