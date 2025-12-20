from typing import Any, Callable
import numpy as np

from .base_loss import BaseLoss
from DepthTensor import Tensor, TensorData, Device, create_2in_1out

EPSILON = 1e-7


def op(x1: TensorData, x2: TensorData, *, device: Device, **kwargs: Any) -> Tensor:
    exps = np.exp(x2 - np.max(x2, axis=-1, keepdims=True))
    y_pred = exps / np.sum(exps, axis=-1, keepdims=True)
    y_pred_clipped = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
    return Tensor(-np.mean(x1 * np.log(y_pred_clipped)), device=device)


def diff(
    result: Tensor, x1: TensorData, x2: TensorData, **kwargs: Any
) -> tuple[Callable[[], TensorData], Callable[[], TensorData]]:
    exps = np.exp(x2 - np.max(x2, axis=-1, keepdims=True))
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    a = 1.0 / x1.size

    def dx1() -> TensorData:
        # dL/d(true) = -log(probs)
        y_pred_clipped = np.clip(probs, EPSILON, 1.0 - EPSILON)
        return result.grad * a * -np.log(y_pred_clipped)  # type: ignore

    def dx2() -> TensorData:
        # dL/d(logits) = (Predictions - Targets)
        return result.grad * a * (probs - x1)  # type: ignore

    return dx1, dx2


func = create_2in_1out(op, diff)


class SoftmaxCategoricalCrossentropy(BaseLoss):
    def __init__(self, name: str = "softmax_categorical_crossentropy") -> None:
        super().__init__(name)

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return super().__call__(y_true, y_pred)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return func(y_true, y_pred)
