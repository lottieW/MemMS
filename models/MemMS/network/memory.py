import torch
from torch import nn
import torch.nn.functional as F

class NormalMemory(nn.Module):
    """Fixed-capacity memory of difficult normal training windows and their routes."""
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.register_buffer('keys', torch.zeros(c.normal_memory_size, c.d_model))
        self.register_buffer('values', torch.zeros(c.normal_memory_size, c.input_channels, c.num_experts))
        self.register_buffer('importance', torch.zeros(c.normal_memory_size))
        self.register_buffer('count', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def reset(self):
        self.keys.zero_()
        self.values.zero_()
        self.importance.zero_()
        self.count.zero_()

    def query(self, z):
        n = int(self.count)
        if n == 0:
            return z.new_full((len(z), self.c.input_channels, self.c.num_experts), 1/self.c.num_experts), z.new_zeros(len(z)), z.new_zeros(len(z))
        # Clone prevents buffer updates from invalidating saved autograd tensors.
        sim = F.normalize(z, dim=-1) @ self.keys[:n].detach().clone().T
        scores, ids = sim.topk(min(n, self.c.memory_topk), dim=-1)
        prior = self.values[:n].detach().clone()[ids].mean(1)
        similarity = scores[:, 0].clamp(-1, 1)
        confidence = similarity.clamp(0, 1)
        return prior, confidence, 1-similarity

    @torch.no_grad()
    def update(self, z, routes, error):
        # Training data are normal: high reconstruction error means a hard normal.
        finite = torch.isfinite(z).all(-1) & torch.isfinite(routes).all((-1, -2)) & torch.isfinite(error)
        eligible = torch.where(finite)[0]
        if not len(eligible):
            return
        # Raw MSE is comparable across batches within an epoch refresh.
        importance = error
        ids = eligible[importance[eligible].argsort(descending=True)]
        for i in ids:
            key = F.normalize(z[i], dim=-1)
            n = int(self.count)
            if n:
                sim, nearest = (self.keys[:n] @ key).max(0)
                # Near-identical keys occupy one slot; keep the more important sample.
                if sim > 0.98:
                    j = int(nearest)
                    if importance[i] > self.importance[j]:
                        self.keys[j] = key
                        self.values[j] = routes[i]
                        self.importance[j] = importance[i]
                    continue
            if n < len(self.keys):
                j = n
                self.count.add_(1)
            else:
                j = int(self.importance.argmin())
                if importance[i] <= self.importance[j]:
                    continue
            self.keys[j] = key
            self.values[j] = routes[i]
            self.importance[j] = importance[i]
