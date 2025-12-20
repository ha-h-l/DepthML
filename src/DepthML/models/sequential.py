from __future__ import annotations
from typing import Any, Sequence

from DepthTensor import Tensor
from .base_model import BaseModel
from ..block import Block
from ..layers import Input, BaseLayer
from ..typing import tensor_types


class Sequential(BaseModel):
    def __init__(
        self, *layers, name: str = "sequential", device: tensor_types.Device = "cpu"
    ) -> None:
        super().__init__(name=name)
        self.layers: list[Block] = []
        self.device = device

        if layers:
            for layer in layers:
                self.add(layer)

            first_layer = self.layers[0]
            if isinstance(first_layer, Input):
                start_shape = first_layer.compute_output_shape()
                self.build(start_shape, device)

    ###
    ### Block abstracts
    ###

    def __call__(self, X: Tensor, **kwargs: Any) -> Tensor:
        return super().__call__(X, **kwargs)

    def call(self, X: Tensor, **kwargs: Any) -> Tensor:
        for layer in self.layers:
            X = layer(X, **kwargs)
        return X

    def build(
        self,
        input_shape: tuple[int, ...],
        device: tensor_types.Device,
        **kwargs: Any,
    ) -> None:
        if device != self.device:
            raise RuntimeError(
                f"Device must match the set device. Expected {self.device}, got {device}."
            )

        for layer in self.layers:
            layer.build(input_shape, device=device)
            input_shape = layer.compute_output_shape(
                input_shape
            )  # input_shape for next layer
        self.built = True

    def compute_output_shape(
        self, input_shape: tuple[int, ...], **kwargs
    ) -> tuple[int, ...]:
        for layer in self.layers:
            input_shape = layer.compute_output_shape(input_shape)
        return input_shape

    ###
    ### Sequential methods
    ###

    def add(self, layer: Block) -> None:
        self.layers.append(layer)

        layer_name = f"{layer.name}{len(self.layers)-1}"
        self._blocks[layer_name] = layer

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx) -> Sequential | Block:
        if isinstance(idx, slice):
            return Sequential(self.layers[idx])
        return self.layers[idx]
