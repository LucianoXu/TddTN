# pylint: disable=line-too-long
from typing import Optional, Any, Sequence, Tuple, Callable, List, Type
from typing import Union
from tensornetwork.backends import abstract_backend
#from tensornetwork.backends.pytorch import decompositions

from . import tdd
import numpy as np

# pylint: disable=abstract-method

Tensor = tdd.TDD

class TDDBackend(abstract_backend.AbstractBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self) -> None:
    super().__init__()
    # pylint: disable=global-variable-undefined
    self.name = "tdd"
  
  def tensordot(self, a: Tensor, b: Tensor,
                axes: Union[int, Sequence[Sequence[int]]]) -> Tensor:
    return tdd.tensordot(a, b, axes=axes)

  def transpose(self, tensor, perm=None) -> Tensor:
    if perm is None:
      perm = tuple(range(tensor.dim_data - 1, -1, -1))
    return tensor.permute(perm)

  
  def shape_tensor(self, tensor: Tensor) -> Tensor:
    return tdd.as_tensor(np.array(tensor.data_shape))
  
  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tuple(tensor.data_shape)
    
  def convert_to_tensor(self, tensor: Any) -> Tensor:
    return tdd.as_tensor(tensor)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return tdd.tensordot(tensor1, tensor2, axes=0)

