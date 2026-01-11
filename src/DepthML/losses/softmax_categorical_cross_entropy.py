from typing import Any
from .base_loss import BaseLoss
from DepthTensor import Tensor, Function
from ..utils import get_xp

import weakref

EPSILON = 1e-7


class SoftmaxCCEFunc(Function):
    def link(self, L: Tensor, y_true: Tensor, y_pred: Tensor, probs: Any) -> None:
        if L.device != y_true.device != y_pred.device:
            raise RuntimeError("There is a mismatch in devices.")
        L_ref = weakref.ref(L)

        def backward() -> None:
            L = L_ref()
            if L is None:
                return

            xp = get_xp(L)
            L_grad = L.grad

            if y_pred.requires_grad:
                if y_pred.grad is None:
                    y_pred.zero_grad()

                batch_size = y_pred.shape[0]
                factor = 1.0 / batch_size

                grad = factor * (probs - y_true.data)
                y_pred.grad += L_grad * grad  # type: ignore

            if y_true.requires_grad:
                if y_true.grad is None:
                    y_true.zero_grad()

                batch_size = y_true.shape[0]
                factor = 1.0 / batch_size

                y_pred_clipped = xp.clip(probs, EPSILON, 1.0 - EPSILON)
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
            raise RuntimeError("There is a mismatch in devices.")
        xp = get_xp(y_pred)

        exps = xp.exp(y_pred.data - xp.max(y_pred.data, axis=-1, keepdims=True))
        probs = exps / xp.sum(exps, axis=-1, keepdims=True)
        y_pred_clipped = xp.clip(probs, EPSILON, 1.0 - EPSILON)
        loss_per_example = -xp.sum(y_true.data * xp.log(y_pred_clipped), axis=-1)

        loss_data = xp.mean(loss_per_example)

        requires_grad = y_pred.requires_grad or y_true.requires_grad
        L = Tensor(loss_data, requires_grad=requires_grad)

        if requires_grad:
            self.link(L, y_true, y_pred, probs)

        return L


softmax_cce_op = SoftmaxCCEFunc()


class SoftmaxCategoricalCrossentropy(BaseLoss):
    def __init__(self, name: str = "softmax_categorical_crossentropy") -> None:
        super().__init__(name)

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return softmax_cce_op(y_true, y_pred)
