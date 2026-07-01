import torch
import torch.optim as optim

class Muon(optim.Optimizer):
    """ Muon - Momentum Orthogonalized by Newton-schulz """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr, momentum, nesterov = group['lr'], group['momentum'], group['nesterov']

            for p in group['params']:
                if p.grad is None or p.grad.ndim == 0:
                    continue

                g = p.grad
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if nesterov else buf.clone()

                # Newton-Schulz orthogonalization for 2D+ tensors
                if g.ndim >= 2:
                    orig_shape = g.shape
                    g = g.view(orig_shape[0], -1)
                    g = g / (g.norm() + 1e-8)

                    a, b, c = (3.4445, -4.7750, 2.0315)
                    for _ in range(5):
                        if g.size(0) < g.size(1):
                            A = g @ g.T
                            g = a * g + b * A @ g + c * A @ A @ g
                        else:
                            A = g.T @ g
                            g = a * g + b * g @ A + c * g @ A @ A

                    g = g.view(orig_shape)

                p.add_(g, alpha=-lr)

        return loss