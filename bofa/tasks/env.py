import math

import torch

from bofa.tasks.objective import Objective

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# See original code for task here: https://github.com/wjmaddox/mtgp_sampler/blob/master/hogp_experiments/data.py
class EnvObjective(Objective):
    """Environmental Pollutants Task Described in
    section 4.4 of BO w/ High-Dimensional Outputs
    paper (https://arxiv.org/pdf/2106.12997.pdf)
    Same as Env objective from BOCF paper, modified
    to be higher-dimensional
    """

    def __init__(
        self,
        dtype=torch.float32,
        **kwargs,
    ):
        M0 = torch.tensor(10.0).to(dtype=dtype).to(device)
        D0 = torch.tensor(0.07).to(dtype=dtype).to(device)
        L0 = torch.tensor(1.505).to(dtype=dtype).to(device)
        tau0 = torch.tensor(30.1525).to(dtype=dtype).to(device)

        self.s_size = 3
        self.t_size = 4
        S = torch.tensor([0.0, 1.0, 2.5]).to(dtype=dtype).to(device)
        T = torch.tensor([15.0, 30.0, 45.0, 60.0]).to(dtype=dtype).to(device)

        self.Sgrid, self.Tgrid = torch.meshgrid(S, T)
        self.c_true = self.env_cfun(self.Sgrid, self.Tgrid, M0, D0, L0, tau0)
        # Bounds used to unnormalize x (optimize in 0 to 1 range for all)
        self.lower_bounds = [7.0, 0.02, 0.01, 30.010]
        self.upper_bounds = [13.0, 0.12, 3.00, 30.295]
        super().__init__(
            dim=4,
            lb=0,
            ub=1,
            dtype=dtype,
            **kwargs,
        )

    def env_cfun(self, s, t, M, D, L, tau):
        c1 = M / torch.sqrt(4 * math.pi * D * t)
        exp1 = torch.exp(-(s**2) / 4 / D / t)
        term1 = c1 * exp1
        c2 = M / torch.sqrt(4 * math.pi * D * (t - tau))
        exp2 = torch.exp(-((s - L) ** 2) / 4 / D / (t - tau))
        term2 = c2 * exp2
        term2[torch.isnan(term2)] = 0.0
        return term1 + term2

    def f(self, x):
        self.num_calls += 1
        x = x.squeeze().to(device)
        # Unnormalize each dim of x
        for i in range(4):
            x[i] = (x[i] * (self.upper_bounds[i] - self.lower_bounds[i])) + self.lower_bounds[i]
        # compute h_x
        h_x = self.env_cfun(self.Sgrid, self.Tgrid, *x)  # torch.Size([3, 4])
        sq_diffs = (h_x - self.c_true).pow(2)
        reward = sq_diffs.sum(dim=(-1, -2)).mul(-1.0)
        return reward.item()
