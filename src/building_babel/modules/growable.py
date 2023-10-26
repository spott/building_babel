import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from typing import Callable, List, Optional

from ..types import InitFunc_

class GrowableRMSNorm(nn.Module):
    """
    A growable, function preserving version of RMSNorm.  This follows https://arxiv.org/pdf/2305.02869.pdf
    and has a set of parameters that are 1 for the old dim, and 0 for the new dim, that are
    multipled by x in _norm, ensuring that the new parameters don't immediately overwhelm the
    norm.  The new params start at 0 and gradually increase over the next few training steps (num_iters) until
    they reach 1 (at which point they are ignored)
    """
    def __init__(self, dim: int, eps: float = 1e-5, num_iters: int = 10):
        super().__init__()
        self.dims = [dim]
        self.eps = eps
        self.weight_splits = nn.ParameterDict({"0": nn.Parameter(torch.ones(dim))})
        self.masks = {"0": torch.ones(dim, requires_grad=False)}
        self.mask_frac = 1
        self.num_iters = num_iters
        self.iter = num_iters
    
    def mask_update(self):
        gen = str(len(self.dims) - 1)
        self.masks[gen] += (1 / self.num_iters)
        self.iter += 1
        self.mask_frac = self.dims[-2] + (self.dims[-1] - self.dims[-2]) * self.iter / self.num_iters

    def _norm(self, x):
        # x/sqrt(avg(x^2) + eps)
        if self.iter >= self.num_iters:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        else:
            y = torch.zeros_like(x)
            for i, (prev_dim, dim) in enumerate(zip([0] + self.dims, self.dims)):
                y[...,prev_dim:dim] += x[...,prev_dim:dim] * self.masks[str(i)]
            m = y.pow(2).mean(-1, keepdim=True)
            m /= self.mask_frac
            y = x * torch.rsqrt(m + self.eps)
            self.mask_update()
            return y

    def grow(self, final_dim: int):
        """
        given the final dimension, modify the norm to work for that dimension
        """
        #old_params = self.weight.data
        delta_dim = final_dim - self.dims[-1]
        new_params = torch.ones(delta_dim)
        gen = str(len(self.dims))
        self.weight_splits[gen] = nn.Parameter(new_params)
        self.iter = 0
        self.masks[gen] = torch.zeros(delta_dim)
        self.mask_frac = self.dims[-1] / final_dim
        self.dims.append(final_dim)

    def forward(self, x):
        # we transform to a 32 bit float before being normed to avoid numerical
        # instability
        x = self._norm(x.float()).type_as(x)
        # x_i * w_i / sqrt(avg(x^2) + eps)
        y = torch.zeros_like(x)
        for i, (prev_dim, dim) in enumerate(zip([0] + self.dims, self.dims)):
            y[...,prev_dim:dim] += x[...,prev_dim:dim] * self.weight_splits[str(i)]
        return y

class GrowableEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[torch.Tensor] = None, _freeze: bool = False,
                 device=None, dtype=None) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq,sparse, _weight, _freeze, device, dtype)

    @torch.no_grad()
    def grow(self, new_dim):
        old_emb_weights = self.weight
        new_weights = old_emb_weights.new_empty((self.num_embeddings, new_dim))
        self.weight = nn.Parameter(new_weights)
        self.reset_parameters()
        self.weight[:, :self.embedding_dim] = old_emb_weights
        self.embedding_dim = new_dim
        return

class GrowableLinear(nn.Module):
    """
    A Linear module that is "growable" (has a "grow" method that changes the dim
    of the underlying matrix).  The API is as close to the Linear API as possible,
    with the addition of the "grow" instance method.

    This does two things:
    1) has a grow method that allows for increasing the size of both dims at the same time.
    (single dim growth isn't supported at the moment).
    2) allows for the individual sections of the matrix (the new sections created by growing)
    to be separate parameters used for training.  Thus, after a growth, you can have the old parameter
    retain the learning rate schedule of the old training run, and the new parameters can have a higher
    learning rate schedule.
    """

    def __init__(
        self, in_dim: int, out_dim: int, bias: bool = False, device=None, dtype=None, run_full=False,
    ) -> None:
        assert not bias, "Doesn't support bias at the moment"
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_dims = [in_dim]
        self.out_dims = [out_dim]
        self.run_full = run_full
        # self.multiple = multiple
        # self.full_matrix = torch.empty((dim, dim), requires_grad=True, **factory_kwargs)
        self.weight_splits: nn.ParameterDict = nn.ParameterDict(
            {"0": nn.Parameter(torch.empty((out_dim, in_dim), **self.factory_kwargs))}
        )
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if len(self.weight_splits) != 1:
            raise NotImplementedError
        nn.init.kaiming_uniform_(self.weight_splits["0"], a=math.sqrt(5))

    def get_section(self, generation: int, quadrant: str | int) -> torch.Tensor:
        assert generation < len(self.in_dims)
        prev_in_dim = self.in_dims[generation-1] if generation > 0 else 0
        prev_out_dim = self.out_dims[generation-1] if generation > 0 else 0
        next_in_dim = self.in_dims[generation]
        next_out_dim = self.out_dims[generation]

        quadrant_map = {0: "ul", 1: "ur", 2: "ll", 3: "lr"}
        if isinstance(quadrant, int):
            quadrant = quadrant_map[quadrant]
        
        if generation == 0 and quadrant != "ul":
            raise ValueError("tried to get a quadrant other than 'ul' for gen 0")

        if quadrant == "ul":
            if generation == 0:
                return self.weight_splits["0"]
            else:
                raise NotImplementedError()
            
        return self.weight_splits[str(generation)][quadrant]

    def grow(self, final_in_dim: int, final_out_dim: int, init:InitFunc_|None = None):
        """
        we need to grow the matrix to the given final_dim, then maintain a list of parameters
        that includes a view into the different parts of the matrix (there are 4 of them?)
        our initial go at this will create a new object.  We can create new rectangles in two
        ways, we can keep the original section slices and each new layer gets more sections, or
        we can have each new layer fixed to 3 new sections:
       
           aaabe      aaabe
           aaabe      aaabe
           aaabe  or  aaabe
           cccde      ccczi
           ffffg      fffhg
        
        we will start by doing the first, which is probably the right decisions.
        """
        assert final_in_dim > self.in_dims[-1] + 1, "Must increase in size by more than 1"
        assert final_out_dim > self.out_dims[-1] + 1 or final_out_dim == self.out_dims[-1], "Must increase out size by more than 1, or zero"
        
        n = len(self.in_dims)
        p = nn.ParameterDict(
            {
                "ll": nn.Parameter(
                    torch.zeros(
                        (final_out_dim - self.out_dims[-1], self.in_dims[-1]),
                        **self.factory_kwargs,
                    )
                ),
                "ur": nn.Parameter(
                    torch.zeros(
                        (self.out_dims[-1], final_in_dim - self.in_dims[-1]),
                        **self.factory_kwargs,
                    )
                ),
                "lr": nn.Parameter(
                    torch.eye(
                        final_out_dim - self.out_dims[-1],
                        final_in_dim - self.in_dims[-1],
                        **self.factory_kwargs,
                    )
                ),
            }
        )

        if init is not None:
            with torch.no_grad():
                for param in p.values():
                    init(param)

        self.weight_splits.update({str(n): p})
        self.out_dims.append(final_out_dim)
        self.in_dims.append(final_in_dim)
        # remove the cache
        if hasattr(self, "_full_matrix"):
            delattr(self, "_full_matrix")

    def full_matrix(self):
        """
        returns the full matrix.  This is copied into place. 
        """
        if hasattr(self, "_full_matrix"):
            return self._full_matrix

        self._full_matrix = torch.zeros(
            self.out_dims[-1], self.in_dims[-1], **self.factory_kwargs
        )

        for i, (prev_in_dim, in_dim, prev_out_dim, out_dim) in enumerate(
            zip([0] + self.in_dims, self.in_dims, [0] + self.out_dims, self.out_dims)
        ):
            if i == 0:
                self._full_matrix[
                    prev_out_dim:out_dim, prev_in_dim:in_dim
                ] = self.weight_splits[str(i)]
            else:
                self._full_matrix[
                    prev_out_dim:out_dim, prev_in_dim:in_dim
                ] = self.weight_splits[str(i)]["lr"]
                self._full_matrix[
                    prev_out_dim:out_dim, :prev_in_dim
                ] = self.weight_splits[str(i)]["ll"]
                self._full_matrix[
                    :prev_out_dim, prev_in_dim:in_dim
                ] = self.weight_splits[str(i)]["ur"]

        return self._full_matrix
    
    @torch.no_grad()
    def __setitem__(self, key, value):
        """note that this is indexing non-transposed matrix.  It is largely for testing, rather than having
        any valid usecase.
        """
        try:
            n,m = key
        except TypeError:
            raise KeyError ("key must be a n,m tuple")
        for i, (prev_in_dim, in_dim, prev_out_dim, out_dim) in enumerate(zip([0] + self.in_dims, self.in_dims, [0] + self.out_dims, self.out_dims)):
            # on every iteration, we can rule out the top left section 
            # being possible (as if it was, we would have already got it)
            if m < in_dim and n < out_dim:
                if i == 0 and n < out_dim and m < in_dim:
                    self.weight_splits[str(i)][n,m] = value
                    break
                elif n >= prev_out_dim and m >= prev_in_dim:
                    self.weight_splits[str(i)]['lr'][n - prev_out_dim,m - prev_in_dim] = value
                    break
                elif n >= prev_out_dim and m < prev_in_dim:
                    self.weight_splits[str(i)]['ll'][n - prev_out_dim,m] = value
                    break
                elif n < prev_out_dim and m >= prev_in_dim:
                    self.weight_splits[str(i)]['ur'][n, m-prev_in_dim] = value
                    break
        else:
            raise KeyError("Key not found")
        
        if hasattr(self, "_full_matrix"):
            delattr(self,"_full_matrix")

    def forward(self, x):
        if self.run_full:
            return F.linear(x, self.full_matrix())
        shape = list(x.size())
        shape[-1] = self.out_dims[-1]
        y = torch.zeros(shape).to(x)

        for i, (prev_in_dim, in_dim, prev_out_dim, out_dim) in enumerate(
            zip([0] + self.in_dims, self.in_dims, [0] + self.out_dims, self.out_dims)
        ):
            if i == 0:
                y[..., :out_dim] = F.linear(x[...,:in_dim], self.weight_splits[str(i)])
            else:
                y[..., prev_out_dim:out_dim] += F.linear(x[...,:prev_in_dim], self.weight_splits[str(i)]['ll'])
                y[..., :prev_out_dim] += F.linear(x[...,prev_in_dim:in_dim], self.weight_splits[str(i)]['ur'])
                y[..., prev_out_dim:out_dim] += F.linear(x[...,prev_in_dim:in_dim], self.weight_splits[str(i)]['lr'])
        return y
    

# The following doesn't work because we can't get a proper reference view into
# another tensor... views aren't actually *views* in autograd (they don't share
# autograd state with their parent...)

#     
# class GrowableLinearInPlace(nn.Module):
#     # Argh, this isn't possible.  The views aren't leaf tensors (they aren't true views), and thus
#     # can't be optimized on.
#     def __init__(
#         self, in_dim: int, out_dim: int, bias: bool = False, device=None, dtype=None
#     ) -> None:
#         assert not bias, "doesn't support bias at the moment"
#         super().__init__()
#         self.factory_kwargs = {"device": device, "dtype": dtype}
#         self.in_dims = [in_dim]
#         self.out_dims = [out_dim]
#         self.weights = nn.Parameter(
#             torch.empty((out_dim, in_dim), **self.factory_kwargs)
#         )
#         self.register_parameter("bias", None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
#
#     @torch.no_grad()
#     def grow(self, in_dim: int, out_dim: int, init: bool | Callable = False) -> None:
#         old_weights = self.weights
#         new_weights = torch.eye(out_dim, in_dim)
#         new_weights[: self.out_dims[-1], : self.in_dims[-1]] = old_weights
#         self.weights = nn.Parameter(new_weights)
#         self.out_dims.append(out_dim)
#         self.in_dims.append(in_dim)
#
#     @torch.no_grad()
#     def __setitem__(self, key, value):
#         """note that this is indexing non-transposed matrix.  It is largely for testing, rather than having
#         any valid usecase.
#         """
#         self.weights.data[key] = value
#
#     @torch.no_grad()
#     def get_section(self, generation: int, quadrant: str | int) -> torch.Tensor:
#         assert generation < len(self.in_dims)
#         prev_in_dim = self.in_dims[generation-1] if generation > 0 else 0
#         prev_out_dim = self.out_dims[generation-1] if generation > 0 else 0
#         next_in_dim = self.in_dims[generation]
#         next_out_dim = self.out_dims[generation]
#
#         quadrant_map = {0: "ul", 1: "ur", 2: "ll", 3: "lr"}
#         if isinstance(quadrant, int):
#             quadrant = quadrant_map[quadrant]
#         
#         if generation == 0 and quadrant != "ul":
#             raise ValueError("tried to get a quadrant other than 'ul' for gen 0")
#
#         if quadrant == "ul":
#             if generation == 0:
#                 return self.weights[:next_out_dim, :next_in_dim]
#             else:
#                 return self.weights[:prev_out_dim, :prev_in_dim]
#         elif quadrant == "ur":
#             return self.weights[prev_out_dim:next_out_dim, :prev_in_dim]
#         elif quadrant == "ll":
#             return self.weights[:prev_out_dim, prev_in_dim:next_in_dim]
#         elif quadrant == "lr":
#             return self.weights[prev_out_dim:next_out_dim, prev_in_dim:next_in_dim]
#         else:
#             raise KeyError(f"Unrecognized quadrant {quadrant:s}")
#
#     def get_all_sections(self) -> List[torch.Tensor]:
#         params = [self.get_section(0,0)]
#         for generation in range(1, len(self.in_dims)):
#             params += [self.get_section(generation, i) for i in range(1, 4)]
#
#         return params
#
#     def forward(self, x):
#         return F.linear(x, self.weights, self.bias)