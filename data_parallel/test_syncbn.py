import os

import pytest

import torch
import torch.distributed as dist
import torch.nn as nn
from syncbn import SyncBatchNorm


def init_process(rank, size, fn, inputs, hid_dim, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '25000'
    dist.init_process_group(backend, rank=rank, world_size=size)
    return fn(rank, size, inputs, hid_dim)


def sync_batchnorm(rank, size, inputs, hid_dim, momentum=0.1, eps=1e-5):
    inputs.requires_grad = True
    outputs = SyncBatchNorm(num_features=hid_dim, momentum=momentum, eps=eps)(inputs)

    if size == 1:
        loss = outputs[:outputs.shape[0] // 2].sum()
    elif rank * 2 < size - 1:
        loss = outputs.sum()
    else:
        loss = 0. * outputs.sum()

    loss.backward()

    return outputs.detach(), inputs.grad.detach()


@pytest.mark.parametrize(["num_workers"], [[1], [4]])
@pytest.mark.parametrize(["hid_dim"], [[128], [256], [512], [1024]])
@pytest.mark.parametrize(["batch_size"], [[32], [64]])
def test_batchnorm(num_workers, hid_dim, batch_size):
    torch.random.manual_seed(42)
    inputs = torch.randn(batch_size, hid_dim, dtype=torch.float32)
    sync_inputs = inputs.clone()
    momentum, eps = 0.1, 1e-5

    inputs.requires_grad = True
    gt_out = nn.BatchNorm1d(num_features=hid_dim, momentum=momentum, eps=eps, affine=False)(inputs)
    loss = gt_out[:gt_out.shape[0] // 2].sum()
    loss.backward()
    gt_out, gt_grads = gt_out.detach(), inputs.grad.detach()

    ctx = torch.multiprocessing.get_context("spawn")
    worker_inputs = [sync_inputs[ind:ind + batch_size // num_workers] for ind in range(0, batch_size, batch_size // num_workers)]
    worker_dims = [hid_dim] * num_workers
    worker_size = [num_workers] * num_workers
    worker_fn = [sync_batchnorm] * num_workers
    worker_rank = range(num_workers)

    with ctx.Pool(processes=num_workers) as pool:
        result = pool.starmap(init_process, zip(worker_rank, worker_size, worker_fn, worker_inputs, worker_dims))

    sync_out, sync_grad = torch.cat([out for out, _ in result]), torch.cat([grad for _, grad in result])

    assert torch.allclose(sync_out, gt_out, atol=1e-3, rtol=0)
    assert torch.allclose(sync_grad, gt_grads, atol=1e-3, rtol=0)


