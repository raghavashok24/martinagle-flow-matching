"""
EXPERIMENT 2 -- The entropic bridge by exact alternating projections
====================================================================
(second half of proposal experiment E2; the grid version of the MFM loop)

Question: starting from a reference R, does alternating
    marginal step   (project onto P: match the market)
    martingale step (project onto M: remove free money)
converge to one legitimate model, and which one?  (Theorem 2 / Csiszar)

On a grid both KL projections can be done exactly, with no neural network:
    marginal step   = rescale rows to mu1, then columns to mu2 (Sinkhorn / IPF).
                      full=False does one row+column sweep per pass, which is
                      what the Appendix A table in the notes uses;
                      full=True repeats the sweep until both marginals hold,
                      which is the exact projection Theorem 2 talks about.
                      Both reach the same limit; the pass-by-pass numbers differ;
    martingale step = in each row, tilt by exp(lambda * (y - x)) with lambda
                      chosen so the row's average move is zero.

This is the 'grid-based bridge' baseline.  MFM replaces each step by a
neural approximation, so these numbers are the targets the neural loop is
debugged against (Aarav's Appendix A).

Expected: price 9.375 -> 6.914 -> ... -> 4.641, KL(P*||R) = 0.602.
"""
import numpy as np
from scipy.optimize import brentq
from common import toy
from exp1_mot_bounds_lp import mot_bounds


def marginal_step(P, full=False, mu1=toy.MU1, mu2=toy.MU2, tol=1e-13, max_iter=10_000):
    """Rescale towards {tables with row sums mu1 and column sums mu2}.
    full=True iterates to convergence (the exact KL projection)."""
    P = P.copy()
    for _ in range(max_iter if full else 1):
        P *= (mu1 / P.sum(axis=1))[:, None]          # fix the rows
        P *= (mu2 / P.sum(axis=0))[None, :]          # fix the columns
        if np.abs(P.sum(axis=1) - mu1).max() < tol:  # rows still right? then done
            break
    return P


def martingale_step(P, move=toy.MOVE):
    """KL projection onto {tables whose rows all have zero average move}.
    The projection is an exponential tilt of each row; row sums are kept."""
    Q = np.zeros_like(P)
    for i in range(P.shape[0]):
        row, mv = P[i], move[i]
        tilted_mean = lambda lam: (row * np.exp(lam * mv) * mv).sum()
        lam = 0.0 if abs(tilted_mean(0.0)) < 1e-15 else brentq(tilted_mean, -5, 5, xtol=1e-14)
        w = row * np.exp(lam * mv)
        Q[i] = w * row.sum() / w.sum()
    return Q


def run_bridge(R, n_pass=400, log_at=(1, 2, 3, 4, 5, 8, 20, 400), full=False):
    """Run the alternating loop from R and log both half-steps of chosen passes."""
    P, log = R.copy(), []
    for n in range(1, n_pass + 1):
        P = marginal_step(P, full)
        a = (toy.defect(P), toy.marginal_errors(P)[1], toy.price(P))
        P = martingale_step(P)
        b = (toy.defect(P), toy.marginal_errors(P)[1], toy.price(P))
        if n in log_at:
            log.append((n, a, b))
    return marginal_step(P, full=True), log


def main():
    R = toy.flat_reference()
    print(f"Start at R: defect {toy.defect(R):.2f}, price {toy.price(R):.2f}   (notes: 6.67, 13.33)\n")

    for full, label in ((False, "one row+column sweep per pass (matches Appendix A of the notes)"),
                        (True, "exact marginal projection per pass")):
        P_star, log = run_bridge(R, full=full)
        print(f"Marginal step = {label}")
        print("        after marginal step        after martingale step")
        print("pass    defect   W1(d2)  price     defect   W1(d2)  price")
        for n, a, b in log:
            print(f"{n:>4}    {a[0]:6.3f}  {a[1]:6.3f}  {a[2]:6.3f}    {b[0]:6.3f}  {b[1]:6.3f}  {b[2]:6.3f}")
        print()

    np.set_printoptions(precision=3, suppress=True)
    vmin, vmax, _, _ = mot_bounds(toy.EXOTIC)
    print("Bridge P* (rows 90/100/110, columns 80..120):\n", P_star)
    print(f"\nprice under P*   : {toy.price(P_star):.3f}   (inside [{vmin:.1f}, {vmax:.1f}])")
    print(f"defect of P*     : {toy.defect(P_star):.2e}")
    print(f"KL(P*   || R)    : {toy.kl(P_star, R):.3f}   (notes: 0.602)")
    print(f"KL(tree || R)    : {toy.kl(toy.tree_table(), R):.3f}   (notes: 0.629)")
    print(f"KL(min  || R)    : {toy.kl(toy.min_table(), R):.3f}   (notes: 1.300)")

    # Sensitivity to the reference: a bell-shaped random-walk R of any width
    # should give the same bridge on the toy (notes, Section 5.1).
    print("\nBridge price under Gaussian random-walk references:")
    for s in (5.0, 10.0, 20.0, 40.0):
        Rg = np.exp(-toy.MOVE ** 2 / (2 * s ** 2)); Rg /= Rg.sum()
        print(f"  spread {s:>4}: {toy.price(run_bridge(Rg, n_pass=400, log_at=())[0]):.3f}")


if __name__ == "__main__":
    main()
