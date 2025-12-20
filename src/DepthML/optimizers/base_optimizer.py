from abc import abstractmethod
from typing import Any
from ..block import Block
from ..typing import tensor_types

from DepthTensor import Tensor


class BaseOptimizer:
    def __init__(self, parameters: list[Tensor], name: str = "optimizer") -> None:
        self.name = name
        self.parameters = parameters

    ###
    ### BaseOptimizer abstracts
    ###

    @abstractmethod
    def zero_grad(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError
