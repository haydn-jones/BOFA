import math
from typing import Optional, Tuple

import fire
import gpytorch
import lightning as L
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

import wandb
from bofa.tasks.rover import RoverObjective
from bofa.turbo_state import (
    TurboState,
    generate_batch,
    get_initial_points,
    update_state,
)

wandb.require("core")


def find_largest_b(N: int) -> Tuple[int, int]:
    x = (-3 + math.sqrt(9 + 4 * (N - 1))) / 2
    x = x + 1
    a = int(math.floor(x))
    b = int(math.floor((N - a) / a))
    return a, b


def main(seed: int, bsz: int, tags: Optional[list[str]] = None):
    if isinstance(tags, str):
        tags = [tags]

    a, b = find_largest_b(bsz)
    BSZ = a + a * b

    wandb.init(project="bofa", tags=tags, config={"seed": seed, "bsz": BSZ})

    L.seed_everything(seed)

    DIM = 60
    DEVICE = torch.device("cuda")
    DTYPE = torch.double
    N_INIT = 20
    MAX_EVALS = 20_000

    obj = RoverObjective(dim=DIM, dtype=DTYPE)

    state = TurboState(dim=DIM, batch_size=BSZ)

    X_turbo = get_initial_points(DIM, N_INIT, device=DEVICE, dtype=DTYPE)
    Y_turbo = obj(X_turbo)

    state = TurboState(DIM, batch_size=BSZ, best_value=Y_turbo.max().item())

    while True:
        train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=DIM,
            )
        )
        model = SingleTaskGP(X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood)

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with gpytorch.settings.max_cholesky_size(float("inf")):
            fit_gpytorch_mll(mll)

            X_next = generate_batch(
                state=state,
                model=model,
                X=X_turbo,
                Y=train_Y,
                batch_size=BSZ,
                acqf="ts",
            )

        Y_next = obj(X_next)

        state = update_state(state=state, Y_next=Y_next)

        X_turbo = torch.cat((X_turbo, X_next), dim=0)
        Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

        if state.restart_triggered:
            state = TurboState(DIM, BSZ, best_value=Y_turbo.max().item())

        print(f"{len(X_turbo)}) Best value: {state.best_value:.3f}, TR length: {state.length:.3f}")
        wandb.log(state.as_dict(), step=len(X_turbo))

        if len(X_turbo) >= MAX_EVALS:
            break

    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
