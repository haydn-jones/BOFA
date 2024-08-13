import math

import torch

from bofa.tasks.objective import Objective

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LangermannObjective(Objective):
    """Langermann optimization task, original from
    https://www.sfu.ca/~ssurjano/langer.html,
    Designed as a minimization task so we multiply by -1
    to obtain a maximization task
    """

    def __init__(
        self,
        d=2,
        m=5,
        dtype=torch.float32,
        **kwargs,
    ):
        # defualt dims
        assert d % 2 == 0
        assert m % 5 == 0
        self.d = d
        self.m = m
        self.A = torch.tensor([[3, 5, 2, 1, 7], [5, 2, 1, 4, 9]]).to(dtype=dtype)  # (2,5) default
        # A = (2,5) by default, need A = (d, m)
        self.A = self.A.repeat(d // 2, m // 5)  # (d, m)
        self.A = self.A.to(device)
        # c = (5,) by default, need c = (m,)
        self.c = torch.tensor([1, 2, 5, 2, 3]).to(dtype=dtype)  # (5,)
        self.c = self.c.repeat(
            m // 5,
        )  # (m,)
        self.c = self.c.to(device)
        # grab PI
        self.pi = math.pi

        super().__init__(
            dim=d,
            lb=0,
            ub=10,
            dtype=dtype,
            **kwargs,
        )

    def f(self, x):
        x = x.to(device)
        x_repeated = torch.cat([x.reshape(self.d, 1)] * self.m, -1)
        h_x = ((x_repeated - self.A) ** 2).sum(0)
        h_x = h_x.reshape(1, -1)
        reward = self.c * torch.exp((-1 * h_x) / self.pi) * torch.cos(self.pi * h_x)
        reward = reward.sum()
        self.num_calls += 1
        return reward.item()
