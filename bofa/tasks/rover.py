import torch
from botorch.utils.transforms import unnormalize

from bofa.tasks.objective import Objective
from bofa.tasks.utils.rover_utils import ConstantOffsetFn, create_large_domain


class RoverObjective(Objective):
    """Rover optimization task
    Goal is to find a policy for the Rover which
    results in a trajectory that moves the rover from
    start point to end point while avoiding the obstacles,
    thereby maximizing reward
    """

    def __init__(
        self,
        dim: int = 60,
        dtype: torch.dtype = torch.float32,
    ):
        assert dim % 2 == 0
        lb: float = -0.5 * 4 / dim
        ub = 4 / dim

        # create rover domain
        self.domain = create_large_domain(n_points=dim // 2)
        # create rover oracle
        f_max = 5.0  # default
        self.oracle = ConstantOffsetFn(self.domain, f_max)

        super().__init__(
            dim=dim,
            lb=lb,
            ub=ub,
            dtype=dtype,
        )

    def f(self, x: torch.Tensor) -> float:
        bounds = torch.stack([torch.full_like(x, self.lb), torch.full_like(x, self.ub)], dim=0)
        x = unnormalize(x, bounds)

        self.num_calls += 1
        reward = torch.tensor(self.oracle(x.cpu().numpy())).to(x)
        return reward.item()
