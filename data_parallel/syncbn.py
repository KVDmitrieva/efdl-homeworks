import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_var, eps: float, momentum: float, training: bool):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        if not training:
            norm_input = (input - running_mean) / torch.sqrt(running_var + eps)
            return norm_input, running_mean, running_var

        n = input.shape[1]

        batch_size = torch.tensor([input.shape[0]]).to(input.device)
        batch_sum = input.sum(dim=0)
        square_sum = torch.sum(input ** 2, dim=0)

        reduce_tensors = torch.cat([batch_size, batch_sum, square_sum])
        dist.all_reduce(reduce_tensors, op=dist.ReduceOp.SUM)

        m = reduce_tensors[0]
        mean = reduce_tensors[1:1 + n] / m

        var = reduce_tensors[1 + n:] / m - mean ** 2
        sqrt_var = torch.sqrt(var + eps)

        norm_input = (input - mean) / sqrt_var

        running_mean = (1 - momentum) * running_mean + momentum * mean
        running_var = (1 - momentum) * running_var + momentum * m * var / (m - 1)

        ctx.save_for_backward(input - mean, sqrt_var, m)
        ctx.mark_non_differentiable(running_mean, running_var)
        return norm_input, running_mean, running_var

    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!

        z, sqrt_var, m = ctx.saved_tensors
        n = z.shape[1]
        t = 1 / sqrt_var

        reduce_tensors = torch.cat([grad_output, grad_output * z])
        dist.all_reduce(reduce_tensors, op=dist.ReduceOp.SUM)

        grad_output_sum = reduce_tensors[:n]
        dt = reduce_tensors[n:]

        grad_input = (grad_output * m - dt * z * (t ** 2) - grad_output_sum) * t / m
        return grad_input, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        # your code here
        self.running_mean = torch.zeros((num_features,))
        self.running_var = torch.ones((num_features,))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above
        norm_input, self.running_mean, self.running_var = sync_batch_norm.apply(
            input, self.running_mean, self.running_var, self.eps, self.momentum, self.training
        )
        return norm_input

