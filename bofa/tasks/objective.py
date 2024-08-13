# Parent class for differnet Objectives/ tasks
from typing import Union

import numpy as np
import torch
from torch import Tensor


class Objective:
    """Base class for any optimization task
    class supports oracle calls and tracks
    the total number of oracle class made during
    optimization
    """

    def __init__(
        self,
        dim: int,
        lb: float,
        ub: float,
        num_calls: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        # search space dim
        self.dim = dim
        # absolute upper and lower bounds on search space
        self.lb = lb
        self.ub = ub
        self.dtype = dtype

    def __call__(self, xs: Union[Tensor, np.ndarray]) -> Tensor:
        """Function defines batched function f(x) (the function we want to optimize).

        Args:
            xs (enumerable): (bsz, dim) enumerable tye of length equal to batch size (bsz), each item in enumerable type must be a float tensor of shape (dim,) (each is a vector in input search space).

        Returns:
            tensor: (bsz, 1) float tensor giving reward obtained by passing each x in xs into f(x).

        """
        if isinstance(xs, np.ndarray):
            xs = torch.from_numpy(xs).to(dtype=self.dtype)

        ys = []
        for x in xs:
            ys.append(self.f(x))

        return torch.tensor(ys).to(dtype=self.dtype).unsqueeze(-1)

    def f(self, x: Tensor) -> float:
        """Function f defines function f(x) we want to optimize. This method should also increment self.num_calls by one.

        Args:
            x (tensor): (1, dim) float tensor giving vector in input search space.

        Returns:
            float: reward value obtained by evaluating f at x

        """
        raise NotImplementedError("Must implement f() specific to desired optimization task")
