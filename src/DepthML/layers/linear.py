from typing import Any
from .base_layer import BaseLayer
from DepthTensor import Tensor
from ..typing import tensor_types, InitializerLike
from ..initializers import GlorotUniform, Zeros

import DepthTensor as DTensor


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
        return super().__call__(X, **kwargs)

    def call(self, X: Tensor, **kwargs: Any) -> Tensor:
        return X @ self.w + self.b

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
        """
        input_shape: (batch_size, ..., input_size)
        output_shape: (batch_size, ..., units)
        """
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
        """
        input_shape: (batch_size, ..., input_size)
        """
        self.w = self.weight_initializer(
            shape=(input_shape[-1], self.units), device=device, requires_grad=True
        )
        self.b = self.bias_initializer(
            shape=(self.units,), device=device, requires_grad=True
        )
