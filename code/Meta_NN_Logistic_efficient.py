from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Meta-learning neural network architecture for the Logistic Case (binary outcome) 

    The Neural net is given synthetic input data 
        - with varying statistical parameters such as noise levels, size, number of predictors, covariance level etc.
    The network learns a set of weights that best generalizes from training to test data
        - this set of weights can represent a heuristic (ML regularizers) or other strategy 
        - we want to inspect the weight representations for functional form at the end 
        - Are compressive nonlinearities performing better than ground truth weights? 

"""

# Hyperparameters
# -----------------------------
@dataclass
class Config:
    out_dir: str = "./outputs/logistic_meta"
    seed: int = 123

    # Training
    epochs: int = 100
    steps_per_epoch: int = 1800          # e.g. 900_000 / 500
    eval_steps: int = 200
    batch_size: int = 500
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Network
    hidden: int = 200

    # Task/data
    nsamp: int = 20
    npred: int = 4
    noise_grid: tuple = tuple(np.linspace(0, 1, 6))
    earlynoise: bool = True
    latenoise: bool = False
    cov_level: float = 0.8
    slope: float = 1.0                   #  Beta_1 = 1/slope
    negative_weights: bool = True

    # Numerics
    dtype: str = "float32"               # "float32" recommended
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Optional: ridge stabilizer for beta_hat fit
    ridge: float = 0.0


def torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_str}")



# covariance function
# -----------------------------
def make_cov(npred: int, level: float, device: str, dtype: torch.dtype) -> torch.Tensor:
    cov = torch.full((npred, npred), float(level), device=device, dtype=dtype)
    cov.fill_diagonal_(1.0)
    return cov


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Synthetic data batch generator
# -----------------------------
class TaskBatchGenerator:
    """
    Generates a batch of meta-learning tasks (datasets), now in a vectorized way.

    Each task:
      - sample ground-truth weights w_true
      - sample train X, generate train Y, add noise
      - estimate empirical beta_hat from (X_train, Y_train)
      - sample test X, compute true probabilities p_test from clean Y_test
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = torch_dtype(cfg.dtype)

        cov = make_cov(cfg.npred, cfg.cov_level, self.device, self.dtype)
        self.chol = torch.linalg.cholesky(cov)  # used to sample correlated normals

        self.beta1 = 1.0 / float(cfg.slope)
        self.beta2 = 0.0

        self.d_in = (cfg.nsamp * cfg.npred) + cfg.nsamp

    @torch.no_grad()
    def _sample_correlated_normal(self, batch: int) -> torch.Tensor:
        """
        Return X ~ N(0, cov) with shape [B, nsamp, npred]
        via Z @ chol^T
        """
        z = torch.randn(batch, self.cfg.nsamp, self.cfg.npred, device=self.device, dtype=self.dtype)
        return z @ self.chol.T

    @torch.no_grad()
    def __call__(self, batch: int, s_noise: float, l_noise: float, return_truth: bool = True) -> dict:
        cfg = self.cfg

        # Ground truth weights
        w_true = torch.rand(batch, cfg.npred, device=self.device, dtype=self.dtype)
        if cfg.negative_weights:
            w_true = w_true * torch.sign(torch.randn(batch, cfg.npred, device=self.device, dtype=self.dtype))

        # Train data (clean)
        x_clean = self._sample_correlated_normal(batch)                            # [B, nsamp, npred]
        y_clean = torch.einsum("bsp,bp->bs", x_clean, w_true)                      # [B, nsamp]

        # Noisy train observations
        x_train = x_clean + torch.randn_like(x_clean) * float(s_noise)             # [B, nsamp, npred]
        y_train = y_clean + torch.randn_like(y_clean) * float(l_noise)             # [B, nsamp]

        # Empirical beta_hat (OLS / LSQ)
        # our original: pinv(X^T X) X^T y  (per task).
        # Here: batched least squares, more stable and fast for small npred.
        # Shapes: A=[B, nsamp, npred], B=[B, nsamp, 1] -> sol=[B, npred, 1]
        sol = torch.linalg.lstsq(x_train, y_train.unsqueeze(-1)).solution
        beta_hat = sol.squeeze(-1)                                                 # [B, npred]

        # Optional ridge stabilization
        # if cfg.ridge and cfg.ridge > 0:
        #     # Solve (X^T X + ridge I) beta = X^T y
        #     xt = x_train.transpose(1, 2)                                           # [B, npred, nsamp]
        #     a = xt @ x_train                                                      # [B, npred, npred]
        #     a = a + cfg.ridge * torch.eye(cfg.npred, device=self.device, dtype=self.dtype).unsqueeze(0)
        #     b = xt @ y_train.unsqueeze(-1)                                         # [B, npred, 1]
        #     beta_hat = torch.linalg.solve(a, b).squeeze(-1)

        # Test data (clean)
        x_new = self._sample_correlated_normal(batch)
        y_test_clean = torch.einsum("bsp,bp->bs", x_new, w_true)                   # [B, nsamp]
        p_test = torch.sigmoid(self.beta1 * (y_test_clean - self.beta2))          # [B, nsamp]

        # Noisy test X (your code uses noisy X_test but p_test from clean X_new)
        x_test = x_new + torch.randn_like(x_new) * float(s_noise)

        # Flatten inputs like your original layout
        x_train_flat = torch.cat([x_train.reshape(batch, -1), y_train], dim=1)     # [B, nsamp*npred + nsamp]
        x_test_flat = x_test.reshape(batch, -1)                                    # [B, nsamp*npred]

        out = {
            "x_train_flat": x_train_flat,
            "x_test_flat": x_test_flat,
            "p_test": p_test,
            "beta_hat": beta_hat,
        }
        if return_truth:
            out["w_true"] = w_true
        return out


# -----------------------------
# Model
# -----------------------------
class MetaLogisticMLP(nn.Module):
    """
    MLP maps x_train_flat -> w_hat, then applies w_hat to x_test to produce logits for nsamp points.
    """
    def __init__(self, d_in: int, npred: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, npred, bias=False)

    def forward(self, x_train_flat: torch.Tensor, x_test_flat: torch.Tensor, nsamp: int, npred: int):
        h = F.relu(self.fc1(x_train_flat))
        h = F.relu(self.fc2(h))
        w_hat = torch.tanh(self.fc3(h))  # keeps weights in [-1, 1]

        bsz = x_test_flat.size(0)
        x_test = x_test_flat.view(bsz, nsamp, npred)                  # [B, nsamp, npred]
        logits = torch.einsum("bsp,bp->bs", x_test, w_hat)            # [B, nsamp]
        return logits, w_hat


# -----------------------------
# Training  Step
# -----------------------------
def run_epoch(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    gen: TaskBatchGenerator,
    cfg: Config,
    s_noise: float,
    l_noise: float,
    anchor_batch: dict | None,
    train: bool,
) -> tuple[float, torch.Tensor | None]:
    """
    Returns:
      avg_loss, snap_w_hat (weights for the anchor batch at final step if train and anchor provided)
    """
    model.train(train)
    losses = []
    snap_w = None

    steps = cfg.steps_per_epoch if train else cfg.eval_steps

    for step in range(steps):
        if train and step == 0 and anchor_batch is not None:
            batch = anchor_batch
        else:
            batch = gen(cfg.batch_size, s_noise=s_noise, l_noise=l_noise, return_truth=False)

        logits, w_hat = model(batch["x_train_flat"], batch["x_test_flat"], cfg.nsamp, cfg.npred)

        # Target is p_test (probabilities), so BCE with logits is stable and appropriate:
        loss_matrix = F.binary_cross_entropy_with_logits(logits, batch["p_test"], reduction="none")
        loss = loss_matrix.mean()  # scalar

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if step == 0 and anchor_batch is not None:
                snap_w = w_hat.detach().cpu()

        losses.append(loss.item())

    return float(np.mean(losses)), snap_w


def train_for_noise(cfg: Config, noise_value: float) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = cfg.device
    dtype = torch_dtype(cfg.dtype)

    gen = TaskBatchGenerator(cfg)

    d_in = (cfg.nsamp * cfg.npred) + cfg.nsamp
    model = MetaLogisticMLP(d_in=d_in, npred=cfg.npred, hidden=cfg.hidden).to(device=device, dtype=dtype)

    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Decide noise placement
    s_noise = float(noise_value) if cfg.earlynoise else 0.0
    l_noise = float(noise_value) if cfg.latenoise else 0.0

    # Anchor batch (fixed across epochs) to mimic your snapweights logic
    anchor = gen(cfg.batch_size, s_noise=s_noise, l_noise=l_noise, return_truth=True)

    train_curve = []
    eval_curve = []
    snapweights = torch.empty(cfg.epochs, cfg.batch_size, cfg.npred)

    t0 = time.time()
    for epoch in range(cfg.epochs):
        train_loss, snap_w = run_epoch(
            model=model,
            opt=opt,
            gen=gen,
            cfg=cfg,
            s_noise=s_noise,
            l_noise=l_noise,
            anchor_batch=anchor,
            train=True,
        )

        eval_loss, _ = run_epoch(
            model=model,
            opt=opt,
            gen=gen,
            cfg=cfg,
            s_noise=s_noise,
            l_noise=l_noise,
            anchor_batch=None,
            train=False,
        )

        train_curve.append(train_loss)
        eval_curve.append(eval_loss)

        if snap_w is not None:
            snapweights[epoch] = snap_w

        print(f"[{epoch+1:>3}/{cfg.epochs}] train_loss={train_loss:.6f} eval_loss={eval_loss:.6f}")

    print(f"Done noise={noise_value:.3f} in {(time.time()-t0)/60:.1f} min")

    # Final snapshot on anchor batch (weights at the end)
    model.eval()
    with torch.no_grad():
        logits_final, w_hat_final = model(anchor["x_train_flat"], anchor["x_test_flat"], cfg.nsamp, cfg.npred)

    # Save weight cloud similar to our original: [w_hat | beta_hat | (optionally w_true)]
    cloud = torch.cat([w_hat_final.cpu(), anchor["beta_hat"].cpu(), anchor["w_true"].cpu()], dim=1)
    cloud_path = out_dir / f"Logistic_snapweights_noise{noise_value:.3f}_npred{cfg.npred}_b{cfg.batch_size}_epochs{cfg.epochs}_cov{cfg.cov_level:.2f}.pt"
    torch.save(cloud, cloud_path)

    # Save learning curves
    df = pd.DataFrame({
        "epoch": np.arange(1, cfg.epochs + 1),
        "avg_loss_train": train_curve,
        "avg_loss_eval": eval_curve,
    })
    csv_path = out_dir / f"Logistic_losses_noise{noise_value:.3f}_npred{cfg.npred}_b{cfg.batch_size}_epochs{cfg.epochs}_cov{cfg.cov_level:.2f}.csv"
    df.to_csv(csv_path, index=False)

    # Save snapweights tensor
    snap_path = out_dir / f"Logistic_anchor_snapweights_noise{noise_value:.3f}_npred{cfg.npred}_b{cfg.batch_size}_epochs{cfg.epochs}_cov{cfg.cov_level:.2f}.pt"
    torch.save(snapweights, snap_path)

    # Save config for provenance
    cfg_path = out_dir / f"config_noise{noise_value:.3f}.json"
    cfg_dict = asdict(cfg)
    with open(cfg_path, "w") as f:
        json.dump(cfg_dict, f, indent=2)


def main():
    cfg = Config()
    set_seed(cfg.seed)

    for noise_value in cfg.noise_grid:
        train_for_noise(cfg, float(noise_value))

    print("Finished noise simulations.")


if __name__ == "__main__":
    main()
