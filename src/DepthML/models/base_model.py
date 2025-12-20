from __future__ import annotations
from ..layers import BaseLayer
from ..typing import OptimizerLike


class BaseModel(BaseLayer):
    def __init__(self, name: str = "model") -> None:
        super().__init__(name=name)
        self.optimizer: OptimizerLike | None = None

    ###
    ### BaseModel methods
    ###

    def use(self, optimizer: OptimizerLike) -> BaseModel:
        self.optimizer = optimizer
        return self
