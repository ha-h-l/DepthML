from typing import Any

from .base_activation import BaseActivation
from ..utils import get_xp

from DepthTensor import Tensor, TensorData, Function


class SigmoidFunc(Function):
    def link(self, y: Tensor, x: TensorData) -> None:
        def backward() -> None:
            if isinstance(x, Tensor):
                if x.grad is None:
                    x.zero_grad()
                if y.grad is None:
                    y.zero_grad()

                # dy/dx = y * (1 - y)
                x.grad += y.grad * (y.data * (1.0 - y.data))  # type: ignore

        y.prev = (x,)
        y.backward = backward

    def __call__(self, x: Tensor) -> Tensor:
        xp = get_xp(x)
        y_data = xp.where(
            x >= 0, 1.0 / (1.0 + xp.exp(-x)), xp.exp(x) / (1.0 + xp.exp(x))
        )
        y = Tensor(y_data, requires_grad=x.requires_grad)
        if y.requires_grad:
            self.link(y, x)
        return y


class TanhFunc(Function):
    def link(self, y: Tensor, x: TensorData) -> None:
        def backward() -> None:
            if isinstance(x, Tensor):
                if x.grad is None:
                    x.zero_grad()
                if y.grad is None:
                    y.zero_grad()

                # dy/dx = 1 - y^2
                x.grad += y.grad * (1.0 - y.data**2)  # type: ignore

        y.prev = (x,)
        y.backward = backward

    def __call__(self, x: Tensor) -> Tensor:
        xp = get_xp(x)
        y_data = xp.tanh(x)
        y = Tensor(y_data, requires_grad=x.requires_grad)
        if y.requires_grad:
            self.link(y, x)
        return y


class ReLUFunc(Function):
    def link(self, y: Tensor, x: TensorData) -> None:
        def backward() -> None:
            if isinstance(x, Tensor):
                if x.grad is None:
                    x.zero_grad()
                if y.grad is None:
                    y.zero_grad()

                # dy/dx = 1 if x > 0 else 0
                x.grad += y.grad * (x.data > 0).astype(x.data.dtype)

        y.prev = (x,)
        y.backward = backward

    def __call__(self, x: Tensor) -> Tensor:
        xp = get_xp(x)
        y_data = xp.maximum(x, 0)
        y = Tensor(y_data, requires_grad=x.requires_grad)
        if y.requires_grad:
            self.link(y, x)
        return y


class LeakyReLUFunc(Function):
    def link(self, y: Tensor, x: TensorData, alpha: float) -> None:
        def backward() -> None:
            if isinstance(x, Tensor):
                if x.grad is None:
                    x.zero_grad()
                if y.grad is None:
                    y.zero_grad()

                xp = get_xp(x.data)
                # dy/dx = 1 if x > 0 else alpha
                grad_mask = xp.where(x.data > 0, 1.0, alpha)
                x.grad += y.grad * grad_mask

        y.prev = (x,)
        y.backward = backward

    def __call__(
        self, x: Tensor, alpha: float = 0.01, differentiate: bool = False
    ) -> Tensor:
        xp = get_xp(x)
        y_data = xp.maximum(x, x * alpha)
        y = Tensor(y_data, requires_grad=x.requires_grad)
        if y.requires_grad:
            self.link(y, x, alpha)
        return y


class ELUFunc(Function):
    def link(self, y: Tensor, x: TensorData, alpha: float) -> None:
        def backward() -> None:
            if isinstance(x, Tensor):
                if x.grad is None:
                    x.zero_grad()
                if y.grad is None:
                    y.zero_grad()

                xp = get_xp(x.data)
                # dy/dx = 1 if x > 0 else alpha * exp(x) (which is y + alpha)
                grad_input = xp.where(x.data > 0, 1.0, y.data + alpha)
                x.grad += y.grad * grad_input

        y.prev = (x,)
        y.backward = backward

    def __call__(
        self, x: Tensor, alpha: float = 1.0, differentiate: bool = False
    ) -> Tensor:
        xp = get_xp(x)
        # y = x if x > 0 else alpha * (exp(x) - 1)
        y_data = xp.where(x > 0, x, alpha * (xp.exp(x) - 1))
        y = Tensor(y_data, requires_grad=x.requires_grad)
        if y.requires_grad:
            self.link(y, x, alpha)
        return y


class GELUFunc(Function):
    """
    Gaussian Error Linear Unit.
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """

    def link(self, y: Tensor, x: TensorData) -> None:
        def backward() -> None:
            if isinstance(x, Tensor):
                if x.grad is None:
                    x.zero_grad()
                if y.grad is None:
                    y.zero_grad()

                xp = get_xp(x.data)
                s = x.data / xp.sqrt(2)
                # Tanh approx derivative:
                tanh_val = xp.tanh(0.7978845608 * (x.data + 0.044715 * x.data**3))

                coeff = 0.5 * (1 + tanh_val)
                inner_deriv = 0.7978845608 * (1 + 3 * 0.044715 * x.data**2)
                sech_sq = 1.0 - tanh_val**2

                dydx = coeff + (0.5 * x.data * sech_sq * inner_deriv)
                x.grad += y.grad * dydx

        y.prev = (x,)
        y.backward = backward

    def __call__(self, x: Tensor) -> Tensor:
        xp = get_xp(x)
        # sqrt(2/pi) ~= 0.7978845608
        inner = 0.7978845608 * (x + 0.044715 * x**3)
        y_data = 0.5 * x * (1.0 + xp.tanh(inner))
        y = Tensor(y_data, requires_grad=x.requires_grad)
        if y.requires_grad:
            self.link(y, x)
        return y


class SoftmaxFunc(Function):
    def link(self, y: Tensor, x: TensorData, axis: int) -> None:
        def backward() -> None:
            if isinstance(x, Tensor):
                if x.grad is None:
                    x.zero_grad()
                if y.grad is None:
                    y.zero_grad()

                xp = get_xp(y.data)
                # y * (grad - sum(grad * y))
                prod = xp.sum(y.grad * y.data, axis=axis, keepdims=True)
                x.grad += y.data * (y.grad - prod)

        y.prev = (x,)
        y.backward = backward

    def __call__(self, x: Tensor, axis: int = -1) -> Tensor:
        xp = get_xp(x)
        x_max = xp.max(x, axis=axis, keepdims=True)
        exp_x = xp.exp(x - x_max)
        y_data = exp_x / xp.sum(exp_x, axis=axis, keepdims=True)
        y = Tensor(y_data, requires_grad=x.requires_grad)
        if x.requires_grad:
            self.link(y, x, axis)
        return y


sigmoid_op = SigmoidFunc()
tanh_op = TanhFunc()
relu_op = ReLUFunc()
leaky_relu_op = LeakyReLUFunc()
elu_op = ELUFunc()
gelu_op = GELUFunc()
softmax_op = SoftmaxFunc()


class Sigmoid(BaseActivation):
    def __call__(self, X: Tensor, **kwargs: Any) -> Tensor:
        return sigmoid_op(X)


class Tanh(BaseActivation):
    def __call__(self, X: Tensor, **kwargs: Any) -> Tensor:
        return tanh_op(X)


class ReLU(BaseActivation):
    def __call__(self, X: Tensor, **kwargs: Any) -> Tensor:
        return relu_op(X)


class LeakyReLU(BaseActivation):
    def __init__(self, alpha: float = 0.01, name: str = "leaky_relu") -> None:
        super().__init__(name)
        self.alpha = alpha

    def __call__(self, X: Tensor, **kwargs: Any) -> Tensor:
        return leaky_relu_op(X, alpha=self.alpha)


class ELU(BaseActivation):
    def __init__(self, alpha: float = 1.0, name: str = "elu") -> None:
        super().__init__(name)
        self.alpha = alpha

    def __call__(self, X: Tensor, **kwargs: Any) -> Tensor:
        return elu_op(X, alpha=self.alpha)


class GELU(BaseActivation):
    def __call__(self, X: Tensor, **kwargs: Any) -> Tensor:
        return gelu_op(X)


class Softmax(BaseActivation):
    def __init__(self, axis: int = -1, name: str = "softmax") -> None:
        super().__init__(name)
        self.axis = axis

    def __call__(self, X: Tensor, **kwargs: Any) -> Tensor:
        return softmax_op(X, axis=self.axis)


__all__ = ["Sigmoid", "Tanh", "ReLU", "LeakyReLU", "ELU", "GELU", "Softmax"]
