import math

import torch

from bofa.tasks.objective import Objective

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LangermannObjectiveAnyDim(Objective):
    """Langermann optimization task, original from
    https://www.sfu.ca/~ssurjano/langer.html,
    adapted to a composite function by
    BO for Composite Functions Paper
    (https://arxiv.org/abs/1906.01537 see Appendix D)
    Designed as a minimization task so we multiply by -1
    to obtain a maximization task
    """

    def __init__(
        self,
        dtype=torch.float32,
        input_dim=None,
        output_dim=None,
        **kwargs,
    ):
        if input_dim is None:
            self.input_dim = self.get_default_input_dim()
        if output_dim is None:
            self.output_dim = self.get_default_output_dim()
        assert self.input_dim % 2 == 0
        assert self.output_dim % 5 == 0
        self.A = torch.tensor([[3, 5, 2, 1, 7], [5, 2, 1, 4, 9]]).to(dtype=dtype)  # (2,5) defaults
        # A = (2,5) by default, need A = (input_dim, output_dim)
        self.A = self.A.repeat(self.input_dim // 2, self.output_dim // 5)  # (input_dim, output_dim)
        self.A = self.A.to(device)
        # A = (5,) by default, need c = (output_dim,)
        self.c = torch.tensor([1, 2, 5, 2, 3]).to(dtype=dtype)  # (5,)
        self.c = self.c.repeat(
            self.output_dim // 5,
        )  # (output_dim,)
        self.c = self.c.to(device)
        # grab PI
        self.pi = math.pi
        # default n init

        super().__init__(
            dim=self.input_dim,
            lb=0,
            ub=10,
            dtype=dtype,
            **kwargs,
        )

    def get_default_input_dim(self):
        return 12  # 60 # 16

    def get_default_output_dim(self):
        return 10  # 60

    def f(self, x):
        x = x.to(device)
        x_repeated = torch.cat([x.reshape(self.dim, 1)] * self.output_dim, -1)
        h_x = ((x_repeated - self.A) ** 2).sum(0)
        h_x = h_x.reshape(1, -1)
        reward = self.c * torch.exp((-1 * h_x) / self.pi) * torch.cos(self.pi * h_x)
        reward = reward.sum().item()
        self.num_calls += 1
        return reward
