from .base_loss import BaseLoss
from DepthTensor import Tensor, Function

import weakref


class MSEFunc(Function):
    def link(self, L: Tensor, y_true: Tensor, y_pred: Tensor) -> None:
        L_ref = weakref.ref(L)

        def backward() -> None:
            L = L_ref()
            if L is None:
                return

            L_grad = L.grad

            # dy_pred = (2/N) * (y_pred - y_true)
            if y_pred.requires_grad:
                if y_pred.grad is None:
                    y_pred.zero_grad()

                n = y_pred.size
                factor = 2.0 / n

                # Gradient calculation
                grad = factor * (y_pred.data - y_true.data)
                y_pred.grad += L_grad * grad  # type: ignore

            # dy_true = (2/N) * (y_true - y_pred)
            if y_true.requires_grad:
                if y_true.grad is None:
                    y_true.zero_grad()

                n = y_true.size
                factor = 2.0 / n

                grad = factor * (y_true.data - y_pred.data)
                y_true.grad += L_grad * grad  # type: ignore

        prev = []
        if y_true.requires_grad:
            prev.append(y_true)
        if y_pred.requires_grad:
            prev.append(y_pred)
        L.prev = tuple(prev)
        L.backward = backward

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        diff = y_true.data - y_pred.data
        loss_data = (diff**2).mean()

        requires_grad = y_pred.requires_grad or y_true.requires_grad
        L = Tensor(loss_data, requires_grad=requires_grad)

        if requires_grad:
            self.link(L, y_true, y_pred)

        return L


mse_op = MSEFunc()


class MeanSquaredError(BaseLoss):
    def __init__(self, name: str = "mean_squared_error") -> None:
        super().__init__(name)

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return mse_op(y_true, y_pred)
