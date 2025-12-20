from __future__ import annotations
from abc import ABC, abstractmethod
from collections import abc, OrderedDict
from typing import Any
from .typing import tensor_types

from DepthTensor import Tensor


class Block(ABC):
    def __init__(self, name: str = "block", trainable: bool = True) -> None:
        super().__init__()
        self._parameters = OrderedDict()
        self._blocks = OrderedDict()

        self.name = name
        self.trainable = trainable
        self.built = False

    @abstractmethod
    def call(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build(
        self,
        input_shape: tuple[int, ...],
        device: tensor_types.Device,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_output_shape(self, input_shape: tuple[int, ...], **kwargs: Any) -> Any:
        raise NotImplementedError

    def parameters(self, include_name: bool = False, prefix: str = "") -> abc.Iterator:
        for name, param in self._parameters.items():
            if include_name:
                yield f"{prefix}{name}", param
            else:
                yield param

        for name, block in self._blocks.items():
            yield from block.parameters(
                include_name=include_name, prefix=f"{prefix}{name}/"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Tensor):
            if not self.trainable:
                raise RuntimeError(
                    "Attempted to set a trainable tensor as an attribute of an untrainable block."
                )
            self._parameters[name] = value
        elif isinstance(value, Block):
            self._blocks[name] = value
        super().__setattr__(name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.built:
            X = args[0]
            self.build(input_shape=X.shape, device=X.device)
        return self.call(*args, **kwargs)
