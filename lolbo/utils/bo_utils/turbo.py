import math
from dataclasses import dataclass
from typing import Tuple

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.generation.sampling import _flip_sub_unique
from botorch.optim import optimize_acqf
from torch import Tensor
from torch.quasirandom import SobolEngine


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 32
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(
    state,
    gp,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=256,
    acqf="ts",  # "ei" or "ts"
    dtype=torch.float32,
    device=torch.device("cuda"),
) -> Tuple[Tensor, Tensor]:
    assert acqf in ("ts", "ei")
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    x_center = X[Y.argmax(), :].clone()
    weights = torch.ones_like(x_center) * 8  # less than 4 stdevs on either side max
    tr_lb = x_center - weights * state.length / 2.0
    tr_ub = x_center + weights * state.length / 2.0

    if acqf == "ei":
        ei = qExpectedImprovement(gp.cuda(), Y.max().cuda())
        X_next, scores = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]).cuda(),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    else:
        dim = X.shape[-1]
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda()
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype).cuda()
        pert = tr_lb + (tr_ub - tr_lb) * pert
        tr_lb = tr_lb.cuda()
        tr_ub = tr_ub.cuda()
        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
        mask = mask.cuda()

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand = X_cand.cuda()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=gp, replacement=False)
        with torch.no_grad():
            X_next, scores = thompson_sampling(X_cand.cuda(), num_samples=batch_size)

    return X_next, scores


class MaxPosteriorSampling(torch.nn.Module):
    def __init__(
        self,
        model,
        replacement: bool = True,
    ) -> None:
        super().__init__()

        self.model = model
        self.replacement = replacement

    def forward(self, X: Tensor, num_samples: int = 1, observation_noise: bool = False) -> Tuple[Tensor, Tensor]:
        posterior = self.model.posterior(X, observation_noise=observation_noise)
        samples = posterior.rsample(sample_shape=torch.Size([num_samples]))
        return self.maximize_samples(X, samples, num_samples)

    def maximize_samples(self, X: Tensor, samples: Tensor, num_samples: int = 1):
        obj = samples.squeeze(-1)
        if self.replacement:
            idcs = torch.argmax(obj, dim=-1)
        else:
            _, idcs_full = torch.topk(obj, num_samples, dim=-1)
            ridx, cindx = torch.tril_indices(num_samples, num_samples)
            sub_idcs = idcs_full[ridx, ..., cindx]
            if sub_idcs.ndim == 1:
                idcs = _flip_sub_unique(sub_idcs, num_samples)
            elif sub_idcs.ndim == 2:
                n_b = sub_idcs.size(-1)
                idcs = torch.stack(
                    [_flip_sub_unique(sub_idcs[:, i], num_samples) for i in range(n_b)],
                    dim=-1,
                )
            else:
                raise NotImplementedError(
                    "MaxPosteriorSampling without replacement for more than a single "
                    "batch dimension is not yet implemented."
                )

        if idcs.ndim > 1:
            idcs = idcs.permute(*range(1, idcs.ndim), 0)

        idcs = idcs.unsqueeze(-1).expand(*idcs.shape, X.size(-1))
        Xe = X.expand(*obj.shape[1:], X.size(-1))
        X_samp = torch.gather(Xe, -2, idcs)
        acq_score = torch.gather(obj, dim=-1, index=idcs[:, :1])

        return X_samp, acq_score
