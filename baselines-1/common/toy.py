"""
Shared toy world for the initial baselines.

This is the two-date toy from the lecture notes (Chapter 3 / Day 2 appendix):
  - today the stock is at 100
  - date 1 grid: 90, 100, 110          with market marginal mu1 = (1/4, 1/2, 1/4)
  - date 2 grid: 80, 90, 100, 110, 120 with market marginal mu2 = (1/16, 1/4, 3/8, 1/4, 1/16)

A two-date path model is a 3x5 table  P[i, j] = Prob(X1 = X1_GRID[i], X2 = X2_GRID[j]).
Everything in experiments 0-3 is a function of such a table, so every number
can be checked by hand against the notes.
"""
import numpy as np

S0 = 100.0
X1_GRID = np.array([90.0, 100.0, 110.0])
X2_GRID = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
MU1 = np.array([1, 2, 1]) / 4.0
MU2 = np.array([1, 4, 6, 4, 1]) / 16.0

# MOVE[i, j] = X2 - X1 for cell (i, j); the running exotic pays |X2 - X1|.
MOVE = X2_GRID[None, :] - X1_GRID[:, None]
EXOTIC = np.abs(MOVE)


# --------------------------------------------------------------------------
# Named tables from the notes
# --------------------------------------------------------------------------
def tree_table():
    """The fair toy tree (also the 'max table': most expensive fair price, 5)."""
    return np.array([[1, 2, 1, 0, 0],
                     [0, 2, 4, 2, 0],
                     [0, 0, 1, 2, 1]]) / 16.0


def min_table():
    """The other extreme fair model (cheapest fair price, 2.5)."""
    return np.array([[0, 4, 0, 0, 0],
                     [1, 0, 6, 0, 1],
                     [0, 0, 0, 4, 0]]) / 16.0


def independent_table():
    """Date 2 drawn independently of date 1: matches the surface, NOT fair."""
    return np.outer(MU1, MU2)


def nonexample_table():
    """Chapter 3 non-example: same mean on every date, but drifts +2.5 / 0 / -2.5.
    From 90 -> 80/90/100 w.p. 1/2,1/4,1/4 ; from 110 -> 100/110/120 w.p. 1/4,1/4,1/2."""
    return np.array([[2, 1, 1, 0, 0],
                     [0, 2, 4, 2, 0],
                     [0, 0, 1, 1, 2]]) / 16.0


def flat_reference():
    """Reference R used in the notes: all 15 (date 1, date 2) pairs equally likely."""
    return np.full((3, 5), 1.0 / 15.0)


# --------------------------------------------------------------------------
# Measurements on a table
# --------------------------------------------------------------------------
def price(P, payoff=EXOTIC):
    """Model price of a payoff given as a 3x5 array (zero rates)."""
    return float((P * payoff).sum())


def drifts(P):
    """Conditional drift m(x1) = E[X2 - X1 | X1 = x1] for each date-1 node."""
    row = P.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (P * MOVE).sum(axis=1) / row
    return np.where(row > 0, m, 0.0)


def defect(P):
    """Martingale defect d(P) = sum_t E|m_t|  (Lemma 3 / v1.0 Lemma 3.2).
    Two terms: the date-0 drift |E[X1] - S0| and the date-1 drifts."""
    row = P.sum(axis=1)
    d0 = abs(float(row @ X1_GRID) - S0)
    d1 = float(np.abs((P * MOVE).sum(axis=1)).sum())   # = sum_x P(x) |m(x)|
    return d0 + d1


def w1(p, q, grid):
    """Wasserstein-1 between two distributions on the same 1-D grid:
    the area between the two cumulative distribution functions."""
    cdf_gap = np.abs(np.cumsum(p - q))[:-1]
    return float((cdf_gap * np.diff(grid)).sum())


def marginal_errors(P):
    """(W1 on date 1, W1 on date 2) against the market marginals."""
    return w1(P.sum(axis=1), MU1, X1_GRID), w1(P.sum(axis=0), MU2, X2_GRID)


def kl(P, R):
    """KL(P || R) = sum P log(P/R); infinite if P puts mass where R has none."""
    mask = P > 0
    if np.any(R[mask] <= 0):
        return float("inf")
    return float((P[mask] * np.log(P[mask] / R[mask])).sum())


def sample_paths(P, n, rng):
    """Draw n paths (X0, X1, X2) from a table. Used by the neural-trader baseline."""
    flat = rng.choice(P.size, size=n, p=P.ravel() / P.sum())
    i, j = np.unravel_index(flat, P.shape)
    return np.stack([np.full(n, S0), X1_GRID[i], X2_GRID[j]], axis=1)
