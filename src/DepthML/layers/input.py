from typing import Tuple, Any
from DepthTensor import Tensor
from ..block import Block
from ..typing import tensor_types


class Input(Block):
    def __init__(self, shape: Tuple[int, ...], name: str = "input") -> None:
        super().__init__(name=name)
        self.input_shape = shape
        self.built = True

    ###
    ### Block abstracts
    ###

    def __call__(self, X: Tensor, **kwargs: Any) -> Tensor:
        return X

    def build(
        self, input_shape: tuple[int, ...], device: tensor_types.Device, **kwargs: Any
    ) -> None:
        pass

    def compute_output_shape(
        self, input_shape: Tuple[int, ...] | None = None, **kwargs
    ) -> Tuple[int, ...]:
        return self.input_shape
