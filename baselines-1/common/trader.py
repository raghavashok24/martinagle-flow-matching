"""
Neural trader = MFM's discriminator (proposal Section 3, step 3a; Lemma 3).

A small network H_phi plays a bounded, predictable trading strategy inside a
set of simulated paths and is trained to maximise its average gain

        gain = mean over paths of  sum_t  H_t * (X_{t+1} - X_t).

Its best gain estimates the martingale defect d(P) from below.

Two rules are enforced here because the notes flag both as easy to get wrong:
  * bounded:      tanh output, so |H_t| <= 1
  * predictable:  H_t sees only X_0..X_t (future prices are masked to zero),
                  never X_{t+1}.  A trader that sees the next price measures
                  volatility, not arbitrage.
"""
import numpy as np
import torch
import torch.nn as nn


class Trader(nn.Module):
    def __init__(self, n_steps, hidden=32):
        super().__init__()
        self.n_steps = n_steps                      # number of trading dates T
        # input: the (masked) path, plus a one-hot of the current date
        self.net = nn.Sequential(
            nn.Linear((n_steps + 1) + n_steps, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh(),        # bounded position in [-1, 1]
        )

    def positions(self, x):
        """x: (batch, T+1) standardised paths.  Returns H of shape (batch, T)."""
        B, T = x.shape[0], self.n_steps
        H = []
        for t in range(T):
            seen = x.clone()
            seen[:, t + 1:] = 0.0                   # predictability: hide the future
            date = torch.zeros(B, T)
            date[:, t] = 1.0
            H.append(self.net(torch.cat([seen, date], dim=1)).squeeze(1))
        return torch.stack(H, dim=1)


def estimate_defect(paths, epochs=300, lr=1e-2, hidden=32, seed=0, holdout=0.5):
    """
    Train a trader on part of the paths and report its gain on the rest.

    paths : array (n, T+1) of simulated prices
    Returns (holdout_gain, train_gain).  The hold-out number is the honest one:
    the training number is biased upward because the trader can fit noise
    (this is the 'Rademacher term' of T-F).
    """
    torch.manual_seed(seed)
    paths = np.asarray(paths, dtype=np.float32)
    n, T = paths.shape[0], paths.shape[1] - 1
    cut = int(n * (1 - holdout))
    dx = torch.tensor(np.diff(paths, axis=1))                # true price moves (dollars)
    scale = paths.std() + 1e-8
    x = torch.tensor((paths - paths[:, :1]) / scale)         # standardised network input

    model = Trader(T, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        gain = (model.positions(x[:cut]) * dx[:cut]).sum(dim=1).mean()
        (-gain).backward()                                   # maximise the gain
        opt.step()

    with torch.no_grad():
        train = (model.positions(x[:cut]) * dx[:cut]).sum(dim=1).mean().item()
        test = (model.positions(x[cut:]) * dx[cut:]).sum(dim=1).mean().item()
    return test, train
