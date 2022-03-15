# pylint: disable=line-too-long
from __future__ import annotations
from typing import Optional, Any, Sequence, Tuple, Callable, List, Type, Union
from typing import Union
from tensornetwork.backends import abstract_backend
#from tensornetwork.backends.pytorch import decompositions


from pytdd import TDD, GlobalOrderCoordinator
import numpy as np

# pylint: disable=abstract-method

Tensor = TDD


class TDDBackend(abstract_backend.AbstractBackend):
  """See base_backend.BaseBackend for documentation."""

  def __init__(self) -> None:
    super().__init__()
    # pylint: disable=global-variable-undefined
    self.coordinator = GlobalOrderCoordinator()
    self.name = "pytdd - global order coordinator"
  
  def tensordot(self, a: Tensor, b: Tensor,
                axes: Union[int, Sequence[Sequence[int]]]) -> Tensor:
    print()
    print("a: ",a.tensor.storage_order)
    print("b: ",b.tensor.storage_order)
    print(axes)
    res = self.coordinator.tensordot(a, b, axes)
    return res

  def transpose(self, tensor, perm=None) -> Tensor:
    if perm is None:
      perm = tuple(range(tensor.dim_data - 1, -1, -1))
    return self.coordinator.permute(tensor,perm)

  def shape_concat(self, values: Tuple[Optional[int],...], axis: int) -> Tuple[Optional[int],...]:
    return np.concatenate(values, axis)

  
  def shape_tensor(self, tensor: Tensor) -> Tuple[Optional[int],...]:
    return tuple(tensor.shape)
  
  def shape_tuple(self, tensor: Tensor) -> Tuple[Optional[int], ...]:
    return tuple(tensor.shape)
    
  def shape_prod(self, values: Tuple[Optional[int], ...]) -> int:
    return np.prod(np.array(values))

  def convert_to_tensor(self, tensor: Any) -> Tensor:
    return self.coordinator.as_tensor(tensor)

  def outer_product(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
    return self.coordinator.tensordot(tensor1, tensor2, 0)

