"""
EXPERIMENT 3 -- Can a neural trader measure free money?
=======================================================
(baseline for the proposal's Lemma 3 / training-loop step 3a)

Question: MFM measures no-arbitrage with a discriminator that trades inside
the simulated paths.  Before using it as a training signal, check that it
recovers the known defect on models where the exact answer is available.

Method: sample paths from each toy model, train the trader of
common/trader.py, report its hold-out gain, and compare with the exact
defect sum_t E|m_t|.

Also included: the mistake the notes warn about.  A trader allowed to see the
NEXT price earns E|X1-X0| + E|X2-X1| = 10 on the fair tree, i.e. it reports
volatility, not arbitrage.  Shown here in closed form as the 'cheating' column.

Expected: tree ~0, independent ~5, non-example ~1.25 (up to sampling noise;
the estimate is a lower bound on the truth plus noise).
"""
import numpy as np
from common import toy
from common.trader import estimate_defect


def cheating_gain(paths):
    """Gain of H_t = sign(X_{t+1} - X_t): what a non-predictable trader would earn."""
    return float(np.abs(np.diff(paths, axis=1)).sum(axis=1).mean())


def main(n_paths=20_000, seed=0):
    rng = np.random.default_rng(seed)
    models = {"tree (fair)": toy.tree_table(),
              "min table (fair)": toy.min_table(),
              "independent": toy.independent_table(),
              "non-example": toy.nonexample_table()}

    print(f"{n_paths} sampled paths per model; half used for training, half held out\n")
    print(f"{'model':<18}{'exact defect':>13}{'trader (held out)':>19}{'trader (train)':>16}{'cheating':>10}")
    for name, P in models.items():
        paths = toy.sample_paths(P, n_paths, rng)
        test, train = estimate_defect(paths, seed=seed)
        print(f"{name:<18}{toy.defect(P):13.3f}{test:19.3f}{train:16.3f}{cheating_gain(paths):10.3f}")

    print("\nRead: 'trader (held out)' should track 'exact defect'.")
    print("      'cheating' is large even for fair models -- never let H_t see X_{t+1}.")


if __name__ == "__main__":
    main()
