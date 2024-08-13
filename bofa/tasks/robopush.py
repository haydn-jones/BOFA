import torch
from botorch.utils.transforms import unnormalize
from torch import Tensor

from bofa.tasks.objective import Objective
from bofa.tasks.utils.robopush_utils import PushReward


class RobotPushing(Objective):
    """14-D Robot Pushing task from TuRBO paper:
    https://arxiv.org/pdf/1910.01739
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        self.robot_push_func = PushReward()

        lb = self.robot_push_func.xmin
        ub = self.robot_push_func.xmax

        super().__init__(
            dim=14,
            dtype=dtype,
            lb=lb,
            ub=ub,
            **kwargs,
        )

    def f(self, x: Tensor) -> float:
        x = x.cpu()
        bounds = torch.stack(
            [
                torch.tensor(self.lb, dtype=self.dtype),
                torch.tensor(self.ub, dtype=self.dtype),
            ]
        )
        x = unnormalize(x, bounds)

        self.num_calls += 1
        y = self.robot_push_func(x.numpy())
        return float(y)
