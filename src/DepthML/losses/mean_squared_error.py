from typing import Any, Callable
from .base_loss import BaseLoss

from DepthTensor import Tensor, TensorData, Device, create_2in_1out


def op(x1: TensorData, x2: TensorData, *, device: Device, **kwargs: Any) -> Tensor:
    return Tensor(((x1 - x2) ** 2).mean(), device=device)


def diff(
    result: Tensor, x1: TensorData, x2: TensorData, **kwargs: Any
) -> tuple[Callable[[], TensorData], Callable[[], TensorData]]:
    a = 2 / x1.size

    def dx1() -> TensorData:
        return result.grad * a * (x1 - x2)  # type: ignore

    def dx2() -> TensorData:
        return result.grad * a * (x2 - x1)  # type: ignore

    return dx1, dx2


func = create_2in_1out(op, diff)


class MeanSquaredError(BaseLoss):
    def __init__(self, name: str = "mean_squared_error") -> None:
        super().__init__(name)

    ###
    ### Block abstracts
    ###

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return super().__call__(y_true, y_pred)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return func(y_true, y_pred)
