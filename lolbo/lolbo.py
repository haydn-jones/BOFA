import math
from typing import Literal, Tuple

import gpytorch
import torch
from gpytorch.mlls import PredictiveLogLikelihood
from torch import Tensor

from lolbo.utils.bo_utils.ppgpr import ApproximateGP, GPModelDKL, inject_post_func
from lolbo.utils.bo_utils.turbo import TurboState, generate_batch, update_state
from lolbo.utils.utils import update_models_end_to_end, update_surr_model

AcqMethod = Literal["vanilla", "half", "quad"]


class LOLBOState:
    def __init__(
        self,
        objective,
        train_x,
        train_y,
        train_z,
        acq_method: AcqMethod,
        k=1_000,
        num_update_epochs=2,
        init_n_epochs=20,
        learning_rte=0.01,
        bsz=10,
        acq_func="ts",
    ):
        self.objective = objective  # objective with vae for particular task
        self.train_x = train_x  # initial train x data
        self.train_y = train_y  # initial train y data
        self.train_z = train_z  # initial train z data
        self.k = k  # track and update on top k scoring points found
        self.num_update_epochs = num_update_epochs  # num epochs update models
        self.init_n_epochs = init_n_epochs  # num epochs train surr model on initial data
        self.learning_rte = learning_rte  # lr to use for model updates
        self.bsz = bsz  # acquisition batch size
        self.acq_func = acq_func  # acquisition function (Expected Improvement (ei) or Thompson Sampling (ts))

        assert acq_func in ["ei", "ts"]

        self.acq_method = acq_method

        self.progress_fails_since_last_e2e = 0
        self.tot_num_e2e_updates = 0
        self.best_score_seen = torch.max(train_y)
        self.best_x_seen = train_x[torch.argmax(train_y.squeeze())]
        self.initial_model_training_complete = (
            False  # initial training of surrogate model uses all data for more epochs
        )
        self.new_best_found = False

        self.initialize_top_k()
        self.initialize_surrogate_model()
        self.initialize_tr_state()
        self.initialize_xs_to_scores_dict()

    def initialize_xs_to_scores_dict(
        self,
    ):
        # put initial xs and ys in dict to be tracked by objective
        init_xs_to_scores_dict = {}
        for idx, x in enumerate(self.train_x):
            init_xs_to_scores_dict[x] = self.train_y.squeeze()[idx].item()
        self.objective.xs_to_scores_dict = init_xs_to_scores_dict

    def initialize_top_k(self):
        """Initialize top k x, y, and zs"""
        # track top k scores found
        self.top_k_scores, top_k_idxs = torch.topk(self.train_y.squeeze(), min(self.k, len(self.train_y)))
        self.top_k_scores = self.top_k_scores.tolist()
        top_k_idxs = top_k_idxs.tolist()
        self.top_k_xs = [self.train_x[i] for i in top_k_idxs]
        self.top_k_zs = [self.train_z[i].unsqueeze(-2) for i in top_k_idxs]

    def initialize_tr_state(self):
        # initialize turbo trust region state
        self.tr_state = TurboState(  # initialize turbo state
            dim=self.train_z.shape[-1], batch_size=self.bsz, best_value=torch.max(self.train_y).item()
        )

        return self

    def initialize_surrogate_model(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
        n_pts = min(self.train_z.shape[0], 1024)
        self.model = GPModelDKL(self.train_z[:n_pts, :].cuda(), likelihood=likelihood).cuda()
        self.mll = PredictiveLogLikelihood(self.model.likelihood, self.model, num_data=self.train_z.size(-2))
        self.model = self.model.eval()
        self.model = self.model.cuda()

        return self

    def update_next(self, z_next_, y_next_, x_next_, acquisition=False):
        """Add new points (z_next, y_next, x_next) to train data
        and update progress (top k scores found so far)
        and update trust region state
        """
        z_next_ = z_next_.detach().cpu()
        y_next_ = y_next_.detach().cpu()
        if len(y_next_.shape) > 1:
            y_next_ = y_next_.squeeze()
        if len(z_next_.shape) == 1:
            z_next_ = z_next_.unsqueeze(0)
        progress = False
        for i, score in enumerate(y_next_):
            self.train_x.append(x_next_[i])
            if len(self.top_k_scores) < self.k:
                # if we don't yet have k top scores, add it to the list
                self.top_k_scores.append(score.item())  # type: ignore
                self.top_k_xs.append(x_next_[i])
                self.top_k_zs.append(z_next_[i].unsqueeze(-2))
            elif score.item() > min(self.top_k_scores) and (x_next_[i] not in self.top_k_xs):  # type: ignore
                # if the score is better than the worst score in the top k list, upate the list
                min_score = min(self.top_k_scores)  # type: ignore
                min_idx = self.top_k_scores.index(min_score)  # type: ignore
                self.top_k_scores[min_idx] = score.item()
                self.top_k_xs[min_idx] = x_next_[i]
                self.top_k_zs[min_idx] = z_next_[i].unsqueeze(-2)  # .cuda()
            # if we imporve
            if score.item() > self.best_score_seen:
                self.progress_fails_since_last_e2e = 0
                progress = True
                self.best_score_seen = score.item()  # update best
                self.best_x_seen = x_next_[i]
                self.new_best_found = True
        if (not progress) and acquisition:  # if no progress msde, increment progress fails
            self.progress_fails_since_last_e2e += 1
        y_next_ = y_next_.unsqueeze(-1)
        if acquisition:
            self.tr_state = update_state(state=self.tr_state, Y_next=y_next_)
        self.train_z = torch.cat((self.train_z, z_next_), dim=-2)
        self.train_y = torch.cat((self.train_y, y_next_), dim=-2)

        return self

    def update_surrogate_model(self):
        if not self.initial_model_training_complete:
            # first time training surr model --> train on all data
            n_epochs = self.init_n_epochs
            train_z = self.train_z
            train_y = self.train_y.squeeze(-1)
        else:
            # otherwise, only train on most recent batch of data
            n_epochs = self.num_update_epochs
            train_z = self.train_z[-self.bsz :]
            train_y = self.train_y[-self.bsz :].squeeze(-1)

        self.model = update_surr_model(self.model, self.mll, self.learning_rte, train_z, train_y, n_epochs)
        self.initial_model_training_complete = True

        return self

    def update_models_e2e(self):
        """Finetune VAE end to end with surrogate model"""
        self.progress_fails_since_last_e2e = 0
        new_xs = self.train_x[-self.bsz :]
        new_ys = self.train_y[-self.bsz :].squeeze(-1).tolist()
        train_x = new_xs + self.top_k_xs
        train_y = torch.tensor(new_ys + self.top_k_scores).float()
        self.objective, self.model = update_models_end_to_end(
            train_x, train_y, self.objective, self.model, self.mll, self.learning_rte, self.num_update_epochs
        )
        self.tot_num_e2e_updates += 1

        return self

    def recenter(self):
        """Pass SELFIES strings back through
        VAE to find new locations in the
        new fine-tuned latent space
        """
        self.objective.vae.eval()
        self.model.train()
        optimizer1 = torch.optim.Adam(
            [{"params": self.model.parameters(), "lr": self.learning_rte}], lr=self.learning_rte
        )
        new_xs = self.train_x[-self.bsz :]
        train_x = new_xs + self.top_k_xs
        max_string_len = len(max(train_x, key=len))
        # max batch size smaller to avoid memory limit
        #   with longer strings (more tokens)
        bsz = max(1, int(2560 / max_string_len))
        num_batches = math.ceil(len(train_x) / bsz)
        for _ in range(self.num_update_epochs):
            for batch_ix in range(num_batches):
                start_idx, stop_idx = batch_ix * bsz, (batch_ix + 1) * bsz
                batch_list = train_x[start_idx:stop_idx]
                z, _ = self.objective.vae_forward(batch_list)
                out_dict = self.objective(z)
                scores_arr = out_dict["scores"]
                valid_zs = out_dict["valid_zs"]
                selfies_list = out_dict["decoded_xs"]
                if len(scores_arr) > 0:  # if some valid scores
                    scores_arr = torch.from_numpy(scores_arr)
                    pred = self.model(valid_zs)
                    loss = -self.mll(pred, scores_arr.cuda())  # type: ignore
                    optimizer1.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer1.step()
                    with torch.no_grad():
                        z = z.detach().cpu()
                        self.update_next(z, scores_arr, selfies_list)
            torch.cuda.empty_cache()
        self.model.eval()

        return self

    def acquisition(self):
        """Generate new candidate points,
        evaluate them, and update data
        """
        if self.acq_method == "vanilla":
            z_next = vanilla_acq(
                tr_state=self.tr_state,
                gp=self.model,
                X=self.train_z,
                Y=self.train_y,
                batch_size=self.bsz,
                acqf=self.acq_func,
            )
        elif self.acq_method == "half":
            a = int(self.bsz / 2)
            b = self.bsz - a

            z_next = split_acq(
                tr_state=self.tr_state,
                gp=self.model,
                X=self.train_z,
                Y=self.train_y,
                a=a,
                b=b,
                acqf=self.acq_func,
            )
        elif self.acq_method == "quad":
            a, b = solve_b(self.bsz)
            z_next = split_acq(
                tr_state=self.tr_state,
                gp=self.model,
                X=self.train_z,
                Y=self.train_y,
                a=a,
                b=b,
                acqf=self.acq_func,
            )
        else:
            raise ValueError("Invalid acquisition method")

        # 2. Evaluate the batch of candidates by calling oracle
        with torch.no_grad():
            out_dict = self.objective(z_next)
            z_next = out_dict["valid_zs"]
            y_next = out_dict["scores"]
            x_next = out_dict["decoded_xs"]

        # 3. Add new evaluated points to dataset (update_next)
        if len(y_next) != 0:
            y_next = torch.from_numpy(y_next).float()
            self.update_next(z_next, y_next, x_next, acquisition=True)
        else:
            self.progress_fails_since_last_e2e += 1
            print("GOT NO VALID Y_NEXT TO UPDATE DATA, RERUNNING ACQUISITOIN...")


def vanilla_acq(
    tr_state: TurboState,
    gp: ApproximateGP,
    X: Tensor,
    Y: Tensor,
    batch_size: int,
    acqf: str,
):
    z_next, scores = generate_batch(
        state=tr_state,
        gp=gp,
        X=X,
        Y=Y,
        batch_size=batch_size,
        acqf=acqf,
    )

    return z_next


def split_acq(
    tr_state: TurboState,
    gp: ApproximateGP,
    X: Tensor,
    Y: Tensor,
    a: int,
    b: int,
    acqf: str,
):
    z_a, scores_a = generate_batch(
        state=tr_state,
        gp=gp,
        X=X,
        Y=Y,
        batch_size=a,
        acqf=acqf,
    )

    try:
        with torch.no_grad():
            fant = gp.variational_strategy.get_fantasy_model(z_a, scores_a.flatten())
            fant = inject_post_func(fant)

        z_b, _ = generate_batch(
            state=tr_state,
            gp=fant,
            X=X,
            Y=Y,
            batch_size=b,
            acqf=acqf,
        )
    except: # Failed to fantasize, just generate the other portion
        z_b, scores_b = generate_batch(
            state=tr_state,
            gp=gp,
            X=X,
            Y=Y,
            batch_size=b,
            acqf=acqf,
        )

    z_next = torch.cat((z_a, z_b), dim=0)

    return z_next


def solve_b(N: int) -> Tuple[int, int]:
    a = math.floor(0.5 * math.sqrt(4 * N + 1) - 1)
    b = N - a
    return a, b
