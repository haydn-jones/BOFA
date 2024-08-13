import math
from dataclasses import asdict, dataclass
from typing import Literal, Optional, Tuple

import torch
from botorch.generation.sampling import _flip_sub_unique
from botorch.models import SingleTaskGP
from torch import Tensor
from torch.quasirandom import SobolEngine


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_tolerance: int = 2
    success_tolerance: int = 3

    length: float = 0.8
    best_value: float = -float("inf")
    restart_triggered: bool = False
    success_counter: int = 0
    failure_counter: int = 0

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))

    def as_dict(self):
        return asdict(self)


def update_state(state: TurboState, Y_next: Tensor) -> TurboState:
    ymax = Y_next.max().item()

    if ymax > state.best_value + 1e-3 * math.fabs(state.best_value):
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

    state.best_value = max(state.best_value, ymax)
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(
    state: TurboState,
    model: SingleTaskGP,
    X: Tensor,  # Evaluated points on the domain [0, 1]^d
    Y: Tensor,  # Function values
    batch_size: int,
    n_candidates: Optional[int] = None,  # Number of candidates for Thompson sampling
    acqf: Literal["ts", "ei"] = "ts",  # Acquisition function to use
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    device = X.device
    dtype = X.dtype

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    dim = X.shape[-1]
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Sample on the candidate points
    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    with torch.no_grad():  # We don't need gradients when using TS
        X_next, _ = thompson_sampling(X_cand, num_samples=batch_size)

    return X_next


def get_initial_points(
    dim: int,
    n_pts: int,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init


class MaxPosteriorSampling(torch.nn.Module):
    def __init__(
        self,
        model: SingleTaskGP,
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
