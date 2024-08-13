import torch

from bofa.tasks.objective import Objective
from bofa.tasks.utils.robopush_utils import PushReward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RobotPushing(Objective):
    """14-D Robot Pushing task from TuRBO paper:
    https://arxiv.org/pdf/1910.01739
    """

    def __init__(
        self,
        dtype=torch.float32,
        **kwargs,
    ):
        self.robot_push_func = PushReward()
        lb = torch.tensor(self.robot_push_func.xmin).to(dtype=dtype)
        ub = torch.tensor(self.robot_push_func.xmax).to(dtype=dtype)

        super().__init__(
            dim=14,
            dtype=dtype,
            lb=lb,
            ub=ub,
            **kwargs,
        )

    def f(self, x):
        self.num_calls += 1
        y = self.robot_push_func(x.numpy())
        return y


if __name__ == "__main__":
    obj = RobotPushing()
    xs = torch.rand(3, obj.dim) * (obj.ub - obj.lb) + obj.lb
    ys = obj(xs)
    print(xs.shape, ys.shape, ys, obj.num_calls)
