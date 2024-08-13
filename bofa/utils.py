from typing import Tuple

from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


def get_turbo_gp(
    X_turbo: Tensor,
    train_Y: Tensor,
    dim: int,
) -> Tuple[SingleTaskGP, ExactMarginalLogLikelihood]:
    likelihood = GaussianLikelihood(noise_constraint=Interval(5e-4, 0.2))
    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=dim,
            lengthscale_constraint=Interval(0.005, 2.0),
        ),
        outputscale_constraint=Interval(0.05, 20),
    )
    model = SingleTaskGP(X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    return model, mll
