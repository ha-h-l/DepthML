from typing import Any

from ..block import Block
from ..typing import tensor_types


class BaseLoss(Block):
    def __init__(self, name: str = "loss") -> None:
        super().__init__(name=name, trainable=False)

    ###
    ### Block abstracts
    ###

    def build(
        self,
        input_shape: tuple[int, ...],
        device: tensor_types.Device = "cpu",
        **kwargs: Any
    ) -> None:
        pass

    def compute_output_shape(
        self, input_shape: tuple[int, ...], **kwargs: Any
    ) -> tuple[int, ...]:
        return ()
