from collections.abc import Iterator
from .base_optimizer import BaseOptimizer
from ..typing import tensor_types

from DepthTensor import Tensor


class SGD(BaseOptimizer):
    def __init__(
        self,
        parameters: list[Tensor] | Iterator,
        learning_rate: float = 0.1,
        name: str = "optimizer",
    ) -> None:
        parameters = parameters if isinstance(parameters, list) else list(parameters)
        super().__init__(parameters=parameters, name=name)
        self.learning_rate = learning_rate

    ###
    ### BaseOptimizer abstracts
    ###

    def zero_grad(self) -> None:
        for param in self.parameters:
            if param.grad is None:
                continue
            param.zeros_grad()

    def step(self) -> None:
        for param in self.parameters:
            if param.grad is None:
                continue
            param.data -= param.grad * self.learning_rate
