from typing import Any
from abc import abstractmethod
from ..block import Block
from ..typing import tensor_types


class BaseLayer(Block):
    def __init__(self, name: str = "layer", trainable: bool = True) -> None:
        super().__init__(name=name, trainable=trainable)
