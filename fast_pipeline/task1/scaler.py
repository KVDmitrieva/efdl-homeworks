import torch


# inspiration: https://on-demand.gputechconf.com/gtc-taiwan/2018/pdf/5-1_Internal%20Speaker_Michael%20Carilli_PDF%20For%20Sharing.pdf

class CustomScaler:
    def __init__(self, mode: str, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=100):
        assert mode in ["static", "dynamic"], "Invalid mode"
        assert growth_factor > 1.0, "Invalid growth factor"
        assert 0.0 < backoff_factor < 1.0, "Invalid backoff factor"

        self._mode = mode
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval

        self._iters_count = 0

    def scale(self, loss):
        return self._scale * loss.float()

    def step(self, optimizer):
        finite_grads = True
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.data.mul_(1. / self._scale)
                    finite_grads = finite_grads and torch.all(torch.isfinite(p.grad))

        if finite_grads:
            optimizer.step()
            self._iters_count += 1
        else:
            self._iters_count = 0

    def update(self):
        if self._mode == "dynamic":
            if self._iters_count == 0:
                self._scale *= self._backoff_factor
            elif self._iters_count == self._growth_interval:
                self._scale *= self._growth_factor
                self._iters_count = 0
