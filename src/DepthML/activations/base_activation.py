from typing import Any
from ..layers import BaseLayer
from ..typing import tensor_types


class BaseActivation(BaseLayer):
    def __init__(self, name: str = "activation") -> None:
        super().__init__(name=name, trainable=False)
        self.built = True

    ###
    ### Block abstracts
    ###

    def build(
        self, input_shape: tuple[int, ...], device: tensor_types.Device, **kwargs: Any
    ) -> None:
        pass

    def compute_output_shape(
        self, input_shape: tuple[int, ...], **kwargs
    ) -> tuple[int, ...]:
        return input_shape
