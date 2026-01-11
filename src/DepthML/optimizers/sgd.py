from collections.abc import Iterator
from .base_optimizer import BaseOptimizer
from DepthTensor import Tensor
from ..utils import get_xp


class SGD(BaseOptimizer):
    def __init__(
        self,
        parameters: list[Tensor] | Iterator,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        name: str = "SGD",
    ) -> None:
        parameters = parameters if isinstance(parameters, list) else list(parameters)
        super().__init__(parameters=parameters, name=name)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def zero_grad(self) -> None:
        for param in self.parameters:
            if param.grad is None:
                continue
            param.zero_grad()

    def step(self) -> None:
        xp = None

        for param in self.parameters:
            if param.grad is None:
                continue

            param_id = id(param)
            if param_id not in self.velocities:
                if xp is None:
                    xp = get_xp(param.data)
                self.velocities[param_id] = xp.zeros_like(param.data)

            v = self.velocities[param_id]
            v *= self.momentum
            v += self.learning_rate * param.grad

            param.data -= v
