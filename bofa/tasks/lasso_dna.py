import LassoBench
import numpy as np
import torch

from bofa.tasks.objective import Objective

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Depends on sparse ho: (automatically instealled using Lassobench setup.py, see below)
#   https://github.com/QB3/sparse-ho/tree/master/sparse_ho

# Carl paper that uses this:
#   https://github.com/hvarfner/vanilla_bo_in_highdim/blob/main/README.md
# LassoBench git
#   https://github.com/ksehic/LassoBench/tree/main


# Do this first:
# git clone https://github.com/ksehic/LassoBench.git
# cd LassoBench/
# pip install -e .
class LassoDNA(Objective):
    """
    https://github.com/ksehic/LassoBench
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.dna_func = LassoBench.RealBenchmark(pick_data="dna")
        super().__init__(
            dim=self.dna_func.n_features,
            lb=-1.0,
            ub=1.0,
            **kwargs,
        )

    def f(self, x):
        self.num_calls += 1
        X_np = x.detach().numpy().flatten().astype(np.float64)
        y = self.dna_func.evaluate(X_np)
        y = y * -1  # negate to create maximization problem

        return y


if __name__ == "__main__":
    obj = LassoDNA()
    xs = torch.rand(3, obj.dim) * (obj.ub - obj.lb) + obj.lb
    ys = obj(xs)
    print(xs.shape, ys.shape, ys, obj.num_calls)
    # dim = 180
    # torch.Size([3, 180]) torch.Size([3, 1])
    #   tensor([[-0.4271],
    #     [-0.4225],
    #     [-0.4130]])
    # 3
