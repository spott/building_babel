"""
In the future, we should look at this:
https://github.com/foundation-model-stack/foundation-model-stack/blob/main/fms/modules/positions.py
(Their implementation doesn't use complex numbers, and can thus be
torch.compiled.)

but initially, we will just do the llama way:

"""

import torch
from logging import getLogger


logger = getLogger("__name__")


class RoPE:
    def __init__(self, dim: int, theta: int = 10_000, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.precompute_freqs_cis(max_seq_len)

    def grow(self, new_dim):
        self.dim = new_dim
        self.precompute_freqs_cis(self.max_seq_len)

    def precompute_freqs_cis(self, end):
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
            Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
            exponentials.
        """
        if self.max_seq_len < end:
            self.max_seq_len = end
        freqs = 1.0 / (
            self.theta
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    def reshape_for_broadcast(self, x: torch.Tensor):
        """
        Reshape frequency tensor for broadcasting it with another tensor.

        This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
        for the purpose of broadcasting the frequency tensor during element-wise operations.

        Args:
            freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
            x (torch.Tensor): Target tensor for broadcasting compatibility.

        Returns:
            torch.Tensor: Reshaped frequency tensor.

        Raises:
            AssertionError: If the frequency tensor doesn't match the expected shape.
            AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert self.freqs_cis.shape[-1] == x.shape[-1]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return self.freqs_cis[: x.shape[1], :].view(*shape)

    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor.

        This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
        frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
        returned as real tensors.

        Args:
            xq (torch.Tensor): Query tensor to apply rotary embeddings.
            xk (torch.Tensor): Key tensor to apply rotary embeddings.
            freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



        """
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self.reshape_for_broadcast(xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
