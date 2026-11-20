"""
EXPERIMENT 4 -- Synthetic truth (Heston) and the naive baselines
================================================================
(first step of proposal experiment E1, 'exact recovery on synthetic truth')

Question: on a market whose true pricing model we know, how do simple
generators score on the two tests and on exotic prices?  This sets the
numbers MFM has to beat, and fixes the evaluation code (per-maturity W1,
measured defect, repricing error) before any MFM model exists.

Ground truth: Heston stochastic volatility, zero rates, simulated by Monte
Carlo.  Its marginals at the maturities play the role of the option surface.

Baselines (all produce prices at the maturities only):
  1. independent : each maturity drawn independently from the true marginal.
                   Matches the surface exactly, badly unfair.
  2. comonotone  : the same quantile on every date (one random number drives
                   the whole path).  Matches the surface exactly, not fair.
  3. black-scholes: constant-volatility random walk with the true terminal
                   variance.  Exactly fair, does not match the surface (no skew).
So 1-2 pass 'market match' and fail 'no free money'; 3 does the opposite.
MFM's target is the row that passes both: 'heston (fresh sample)'.

Metrics
  W1      : sum over maturities of W1(model marginal, true marginal)
  defect  : free money, two estimates --
            'binned'  = sum_t E|E[dX_t | X_t-bin]|, bins fitted on half the
                        paths and evaluated on the other half;
            'trader'  = neural trader that sees the whole past (hold-out gain).
  prices  : call, Asian call, up-and-out call, lookback, cliquet
            (all monitored at the maturities), against the truth.

NOT here yet (next round): rough Bergomi, Dupire local volatility, the
neural-SDE market model, and MFM itself.
"""
import numpy as np
from common.trader import estimate_defect

S0 = 100.0
MATURITIES = np.array([0.25, 0.5, 0.75, 1.0])
HESTON = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
def simulate_heston(n, rng, steps_per_year=252, v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7):
    """Heston paths by Euler with full truncation; zero rates, so S is a martingale.
    Returns an (n, K+1) array: S0 and the price at each maturity."""
    dt = 1.0 / steps_per_year
    n_steps = int(round(MATURITIES[-1] * steps_per_year))
    keep = {int(round(T * steps_per_year)) for T in MATURITIES}
    logS, v, out = np.full(n, np.log(S0)), np.full(n, v0), [np.full(n, S0)]
    for k in range(1, n_steps + 1):
        z1 = rng.standard_normal(n)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        vp = np.maximum(v, 0.0)
        logS += -0.5 * vp * dt + np.sqrt(vp * dt) * z1      # drift-free in S
        v += kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
        if k in keep:
            out.append(np.exp(logS))
    return np.stack(out, axis=1)


# --------------------------------------------------------------------------
# Naive baselines (each takes the true paths and returns paths of the same shape)
# --------------------------------------------------------------------------
def baseline_independent(truth, rng):
    """Shuffle each maturity column separately: marginals kept, dependence destroyed."""
    out = truth.copy()
    for k in range(1, truth.shape[1]):
        out[:, k] = rng.permutation(truth[:, k])
    return out


def baseline_comonotone(truth, rng):
    """Sort every maturity column: path i sits at the same quantile on every date."""
    out = np.sort(truth, axis=0)
    return out[rng.permutation(len(out))]


def baseline_black_scholes(truth, rng):
    """Constant-vol geometric random walk matched to the true terminal log-variance."""
    n = truth.shape[0]
    sigma = np.log(truth[:, -1] / S0).std() / np.sqrt(MATURITIES[-1])
    dt = np.diff(np.concatenate([[0.0], MATURITIES]))
    z = rng.standard_normal((n, len(dt)))
    logret = np.cumsum(-0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * z, axis=1)
    return np.concatenate([np.full((n, 1), S0), S0 * np.exp(logret)], axis=1)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def w1_samples(a, b):
    """W1 between two equal-size 1-D samples: mean gap between sorted values."""
    return float(np.abs(np.sort(a) - np.sort(b)).mean())


def total_w1(model, truth):
    return sum(w1_samples(model[:, k], truth[:, k]) for k in range(1, truth.shape[1]))


def defect_binned(paths, n_bins=50):
    """Markov estimate of the defect: bin X_t by quantile, estimate the drift
    in each bin on the first half of the paths, then trade sign(drift) on the
    second half.  Out-of-sample, so noise does not show up as free money."""
    half = len(paths) // 2
    fit, test = paths[:half], paths[half:]
    total = abs(test[:, 1].mean() - S0)                      # date-0 drift
    for t in range(1, paths.shape[1] - 1):
        edges = np.quantile(fit[:, t], np.linspace(0, 1, n_bins + 1)[1:-1])
        b_fit, b_test = np.digitize(fit[:, t], edges), np.digitize(test[:, t], edges)
        d_fit = fit[:, t + 1] - fit[:, t]
        drift = np.array([d_fit[b_fit == b].mean() if np.any(b_fit == b) else 0.0
                          for b in range(n_bins)])
        total += float((np.sign(drift)[b_test] * (test[:, t + 1] - test[:, t])).mean())
    return total


PAYOFFS = {   # all zero-rate, monitored at the maturities; X has columns S0, S_T1..S_TK
    "call K=100":      lambda X: np.maximum(X[:, -1] - 100, 0),
    "asian call":      lambda X: np.maximum(X[:, 1:].mean(axis=1) - 100, 0),
    "up-out call 120": lambda X: np.maximum(X[:, -1] - 100, 0) * (X[:, 1:].max(axis=1) < 120),
    "lookback":        lambda X: X.max(axis=1) - X[:, -1],
    "cliquet":         lambda X: np.abs(np.diff(X, axis=1)).sum(axis=1),
}


def main(n_paths=100_000, seed=0, use_trader=True):
    rng = np.random.default_rng(seed)
    truth = simulate_heston(n_paths, rng, **HESTON)
    models = {
        "heston (fresh sample)": simulate_heston(n_paths, rng, **HESTON),   # Monte Carlo noise floor
        "independent":           baseline_independent(truth, rng),
        "comonotone":            baseline_comonotone(truth, rng),
        "black-scholes":         baseline_black_scholes(truth, rng),
    }

    print(f"Heston truth: {n_paths} paths, maturities {MATURITIES}, params {HESTON}")
    print(f"mean price at each maturity (should be {S0}): {truth[:, 1:].mean(axis=0).round(3)}\n")

    print("The two tests")
    print(f"{'model':<24}{'sum W1':>9}{'defect (binned)':>17}{'defect (trader)':>17}")
    for name, X in models.items():
        tr = estimate_defect(X[:20_000], epochs=200, seed=seed)[0] if use_trader else float("nan")
        print(f"{name:<24}{total_w1(X, truth):9.3f}{defect_binned(X):17.3f}{tr:17.3f}")

    print("\nRepricing (truth, then each model's price and its error)")
    print(f"{'payoff':<17}{'truth':>8}" + "".join(f"{n.split()[0]:>22}" for n in models))
    for pname, f in PAYOFFS.items():
        ref = f(truth).mean()
        cells = "".join(f"{f(X).mean():>12.3f} ({f(X).mean() - ref:+6.3f})" for X in models.values())
        print(f"{pname:<17}{ref:8.3f}{cells}")

    print("\nRead: independent and comonotone match every vanilla (W1 = 0) and still")
    print("      misprice the path-dependent payoffs; their defect says why.")
    print("      Black-Scholes is fair (defect ~ 0) but misses the surface (W1 > 0).")
    print("      Defects within about +-0.2 of zero are Monte Carlo noise at this sample size;")
    print("      the 'heston (fresh sample)' row is the noise floor for every column.")


if __name__ == "__main__":
    main()
