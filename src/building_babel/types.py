from collections.abc import Callable
from torch import Tensor
from dataclasses import dataclass

InitFunc_ = Callable[[Tensor], Tensor | None]


@dataclass
class TransformerConfig:
    dim: int
    n_layers: int
    vocab_size: int
    norm_eps: float = 1e-5
    head_dim: int = 128
    multiple_of: int = 128
    max_seq_len: int = 4096
    theta: int = 10000
