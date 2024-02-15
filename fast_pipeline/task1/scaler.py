import torch


class CustomScaler:
    def __init__(self, mode: str, init_scale=65536.0):
        assert mode in ["static", "dynamic"], "Invalid mode"
        self._mode = mode
        self._scale_factor = init_scale
        self._need_update = False

    def scale(self, loss):
        return self._scale_factor * loss.float()

    def step(self, optimizer):
        nan_flag = False
        inf_flag = False
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.mul_(1. / self._scale_factor)
                    nan_flag |= torch.any(torch.isnan(p.grad))
                    inf_flag |= torch.any(torch.isinf(p.grad))

        if inf_flag or nan_flag:
            self._need_update = self._mode == "dynamic"
        else:
            optimizer.step()

    def update(self):
        if self._need_update:
            self._scale_factor /= 2.0
            self._need_update = False
