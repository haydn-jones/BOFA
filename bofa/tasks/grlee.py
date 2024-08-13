import math

import torch

from bofa.tasks.objective import Objective

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GramacyLEE(Objective):
    """Gramacy and Lee
    https://www.sfu.ca/~ssurjano/grlee12.html
    Gramacy, R. B., & Lee, H. K. (2012).
    Cases for the nugget in modeling computer experiments.
    Statistics and Computing, 22(3), 713-722.
    """

    def __init__(
        self,
        **kwargs,
    ):
        self.pi_ = math.pi

        super().__init__(
            dim=1,
            lb=0.5,
            ub=2.5,
            **kwargs,
        )

    def f(self, x):
        x = x.item()
        term1 = math.sin(10 * self.pi_ * x) / (2 * x)
        term2 = (x - 1) ** 4
        y = term1 + term2
        self.num_calls += 1
        return -y


if __name__ == "__main__":
    obj = GramacyLEE()
    x = torch.randn(1, 1)
    y = obj(x)
