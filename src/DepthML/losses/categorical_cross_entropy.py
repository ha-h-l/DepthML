from .base_loss import BaseLoss
from DepthTensor import Tensor, Function
from ..utils import get_xp

import weakref

EPSILON = 1e-7


class CCEFunc(Function):
    def link(self, L: Tensor, y_true: Tensor, y_pred: Tensor) -> None:
        if L.device != y_true.device != y_pred.device:
            raise RuntimeError("There is a mismatch between devices.")
        L_ref = weakref.ref(L)

        def backward() -> None:
            L = L_ref()
            if L is None:
                return

            xp = get_xp(L)
            L_grad = L.grad
            y_pred_clipped = xp.clip(y_pred.data, EPSILON, 1.0 - EPSILON)

            # dL/d(pred) = -(y_true / y_pred) * (1/N)
            if y_pred.requires_grad:
                if y_pred.grad is None:
                    y_pred.zero_grad()

                n = y_pred.size
                factor = 1.0 / n

                grad = -factor * (y_true.data / y_pred_clipped)
                y_pred.grad += L_grad * grad  # type: ignore

            # dL/d(true) = -log(y_pred) * (1/N)
            if y_true.requires_grad:
                if y_true.grad is None:
                    y_true.zero_grad()

                n = y_true.size
                factor = 1.0 / n

                grad = -factor * xp.log(y_pred_clipped)
                y_true.grad += L_grad * grad  # type: ignore

        prev = []
        if y_true.requires_grad:
            prev.append(y_true)
        if y_pred.requires_grad:
            prev.append(y_pred)
        L.prev = tuple(prev)
        L.backward = backward

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        if y_true.device != y_pred.device:
            raise RuntimeError("There is a mismatch between device.")
        xp = get_xp(y_pred)

        y_pred_clipped = xp.clip(y_pred.data, EPSILON, 1.0 - EPSILON)
        # L = - sum(y_true * log(y_pred)) / N
        loss_data = -xp.mean(y_true.data * xp.log(y_pred_clipped))

        requires_grad = y_pred.requires_grad or y_true.requires_grad
        L = Tensor(loss_data, requires_grad=requires_grad)

        if requires_grad:
            self.link(L, y_true, y_pred)

        return L


cce_op = CCEFunc()


class CategoricalCrossentropy(BaseLoss):
    def __init__(self, name: str = "categorical_crossentropy") -> None:
        super().__init__(name)

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return cce_op(y_true, y_pred)
