from collections.abc import Iterable

import multiprocess as mp
import numpy as np
import torch

from bofa.tasks.objective import Objective
from bofa.tasks.utils.lunar_utils import simulate_lunar_lander


class LunarLanderObjective(Objective):
    """Lunar Lander optimization task
    Goal is to find a policy for the Lunar Lander
    smoothly lands on the moon without crashing,
    thereby maximizing reward
    """

    def __init__(
        self,
        seed=np.arange(50),
        **kwargs,
    ):
        super().__init__(dim=12, lb=0.0, ub=1.0, **kwargs)
        self.pool = mp.Pool(mp.cpu_count())
        seed = [seed] if not isinstance(seed, Iterable) else seed
        self.seed = seed

    def f(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = x.reshape((-1, self.dim))  # bsz x 12 (1, 12)
        ns = len(self.seed)  # default 50
        nx = x.shape[0]  # bsz = 1 if pass in one policy x at a time
        x_tiled = np.tile(x, (ns, 1))  # ns x dim  (10 seds x 12 dim )
        seed_rep = np.repeat(self.seed, nx)  # repeat ns x number of policies (bsz) = (ns,) when bsz is 1
        params = [[xi, si] for xi, si in zip(x_tiled, seed_rep)]
        # list with pairs of x's and seeds
        # so for a single s, we have a list with [(x, s1), (x,s2), ... (x,sN)]
        # sumulates lunar lander w/ each pair of (x, seed)
        rewards = np.array(self.pool.map(simulate_lunar_lander, params)).reshape(-1)
        # Compute the average score across the seeds
        mean_reward = np.mean(rewards, axis=0).squeeze()
        self.num_calls += 1

        return mean_reward
