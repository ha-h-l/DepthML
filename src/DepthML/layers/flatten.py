from typing import Any
from .base_layer import BaseLayer
from DepthTensor import Shape, Device, Tensor, Function
from ..utils import get_xp

import math


class FlattenFunc(Function):
    def link(self, y: Tensor, x: Tensor) -> None:
        original_shape = x.shape

        def backward() -> None:
            if not y.requires_grad:
                return
            if y.grad is None:
                y.zero_grad()
            if x.requires_grad:
                if x.grad is None:
                    x.zero_grad()

                grad_reshaped = y.grad.reshape(original_shape)  # type: ignore
                x.grad += grad_reshaped  # type: ignore

        if x.requires_grad:
            y.prev = (x,)
        y.backward = backward

    def __call__(self, x: Tensor, differentiate: bool = True) -> Tensor:
        xp = get_xp(x.data)
        if not x.data.flags.c_contiguous:
            x.data = xp.ascontiguousarray(x.data)

        N = x.shape[0]
        out_data = x.data.reshape(N, -1)

        y = Tensor(out_data, requires_grad=x.requires_grad, device=x.device)

        if x.requires_grad and differentiate:
            self.link(y, x)

        return y


flatten_op = FlattenFunc()


class Flatten(BaseLayer):
    def __init__(self, name: str = "layer", trainable: bool = False) -> None:
        super().__init__(name, trainable)

    def __call__(self, X: Tensor, **kwargs: Any) -> Tensor:
        return flatten_op(X)

    def build(self, input_shape: Shape, device: Device, **kwargs: Any) -> None:
        pass

    def compute_output_shape(self, input_shape: Shape, **kwargs: Any) -> Shape:
        return (math.prod(input_shape),)
