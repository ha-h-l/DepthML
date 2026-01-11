from typing import Any

from .base_layer import BaseLayer
from DepthTensor import Tensor, Function
from ..typing import tensor_types, InitializerLike
from ..initializers import GlorotUniform, Zeros
from ..utils import get_xp


class LinearFunc(Function):
    def link(self, y: Tensor, x: Tensor, w: Tensor, b: Tensor | None) -> None:
        def backward() -> None:
            if y.grad is None:
                return

            for p in [x, w, b]:
                if isinstance(p, Tensor) and p.requires_grad and p.grad is None:
                    p.zero_grad()

            dy = y.grad
            xp = get_xp(dy)

            if b is not None and b.requires_grad:
                reduce_dims = tuple(range(dy.ndim - 1))
                b.grad += xp.sum(dy, axis=reduce_dims)

            if w.requires_grad:
                x_flat = x.data.reshape(-1, x.data.shape[-1])
                dy_flat = dy.reshape(-1, dy.shape[-1])
                w.grad += x_flat.T @ dy_flat

            if x.requires_grad:
                x.grad += dy @ w.data.T

        prev = [p for p in [x, w, b] if isinstance(p, Tensor)]
        y.prev = tuple(prev)
        y.backward = backward

    def __call__(self, x: Tensor, w: Tensor, b: Tensor | None) -> Tensor:
        # X @ W + b
        res = x.data @ w.data
        if b is not None:
            res += b.data

        requires_grad = (
            x.requires_grad or w.requires_grad or (b.requires_grad if b else False)
        )
        out = Tensor(res, requires_grad=requires_grad, device=x.device)

        if requires_grad:
            self.link(out, x, w, b)

        return out


linear_op = LinearFunc()


class Linear(BaseLayer):
    def __init__(
        self,
        units: int,
        weight_initializer: InitializerLike = GlorotUniform(),
        bias_initializer: InitializerLike = Zeros(),
        name: str = "linear",
    ) -> None:
        super().__init__(name=name)

        self.units = units
        self.w: Tensor | None = None
        self.b: Tensor | None = None

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

    ###
    ### Block abstracts
    ###

    def __call__(self, X: Tensor, **kwargs: Any) -> Tensor:
        if not self.built:
            self.build(X.shape[1:], X.device)
        return linear_op(X, self.w, self.b)

    def build(
        self,
        input_shape: tuple[int, ...],
        device: tensor_types.Device,
        **kwargs: Any,
    ) -> None:
        self.init_parameters(input_shape=input_shape, device=device)
        self.built = True

    def compute_output_shape(
        self, input_shape: tuple[int, ...], **kwargs
    ) -> tuple[int, ...]:
        return input_shape[:-1] + (self.units,)

    ###
    ###
    ###

    def init_parameters(
        self,
        input_shape: tuple[int, ...],
        device: tensor_types.Device = "cpu",
        **kwargs: Any,
    ) -> None:
        self.w = self.weight_initializer(
            shape=(input_shape[-1], self.units), device=device, requires_grad=True
        )
        self.b = self.bias_initializer(
            shape=(self.units,), device=device, requires_grad=True
        )
