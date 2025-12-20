from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .block import Block
    from .activations.base_activation import BaseActivation
    from .layers.base_layer import BaseLayer
    from .models.base_model import BaseModel
    from .optimizers.base_optimizer import BaseOptimizer
    from .losses.base_loss import BaseLoss
    from .initializers.base_initializer import BaseInitializer

from DepthTensor import typing as tensor_types

BlockLike: TypeAlias = "Block"
ActivationLike: TypeAlias = "BaseActivation"
LayerLike: TypeAlias = "BaseLayer"
ModelLike: TypeAlias = "BaseModel"
OptimizerLike: TypeAlias = "BaseOptimizer"
LossLike: TypeAlias = "BaseLoss"
InitializerLike: TypeAlias = "BaseInitializer"
