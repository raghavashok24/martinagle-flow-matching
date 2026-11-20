"""
EXPERIMENT 1 -- Exact arbitrage-free price bounds by linear programming
=======================================================================
(first half of proposal experiment E2, 'tightness against exact solvers')

Question: given only the option surface, what is the cheapest and the most
expensive price a fair, calibrated model can give the exotic?

Method: classical martingale optimal transport (MOT).  The unknowns are the
15 cells of the table.  All rules are linear:
    rows add up to mu1            (match the market on date 1)
    columns add up to mu2         (match the market on date 2)
    each row has zero average move (no free money)
    every cell >= 0
so min / max of the (linear) price is a linear program.

This is the exact baseline every learned model is compared against: the
certificate T-C measures the distance from a model's price to this interval.
Expected on the toy: [2.5, 5].

It is also the baseline that cannot scale: the number of unknowns is
(grid size)^(number of dates), printed at the end.
"""
import numpy as np
from scipy.optimize import linprog
from common import toy


def mot_bounds(payoff, x1=toy.X1_GRID, x2=toy.X2_GRID, mu1=toy.MU1, mu2=toy.MU2):
    """Return (V_min, V_max, argmin table, argmax table) for a two-date payoff."""
    n1, n2 = len(x1), len(x2)
    idx = lambda i, j: i * n2 + j          # cell (i, j) -> position in the flat vector
    A, b = [], []

    for i in range(n1):                    # row sums = mu1
        r = np.zeros(n1 * n2); r[[idx(i, j) for j in range(n2)]] = 1
        A.append(r); b.append(mu1[i])
    for j in range(n2):                    # column sums = mu2
        r = np.zeros(n1 * n2); r[[idx(i, j) for i in range(n1)]] = 1
        A.append(r); b.append(mu2[j])
    for i in range(n1):                    # fairness: sum_j P[i,j] (x2_j - x1_i) = 0
        r = np.zeros(n1 * n2)
        for j in range(n2):
            r[idx(i, j)] = x2[j] - x1[i]
        A.append(r); b.append(0.0)

    A, b, c = np.array(A), np.array(b), payoff.ravel()
    lo = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    hi = linprog(-c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    assert lo.success and hi.success, "LP infeasible: marginals not in convex order?"
    return lo.fun, -hi.fun, lo.x.reshape(n1, n2), hi.x.reshape(n1, n2)


def main():
    vmin, vmax, Pmin, Pmax = mot_bounds(toy.EXOTIC)
    print(f"Arbitrage-free interval for |X2 - X1|: [{vmin:.4f}, {vmax:.4f}]   (notes: [2.5, 5])")
    np.set_printoptions(precision=4, suppress=True)
    print("\nCheapest fair table (x16):\n", Pmin * 16)
    print("\nMost expensive fair table (x16):\n", Pmax * 16)

    # Where the named models sit relative to the interval.
    print("\nDistance of each named model's price to the interval:")
    for name, P in {"tree": toy.tree_table(), "independent": toy.independent_table(),
                    "non-example": toy.nonexample_table()}.items():
        p = toy.price(P)
        dist = max(vmin - p, 0, p - vmax)
        print(f"  {name:<12} price {p:6.3f}   distance {dist:6.3f}")

    # A second payoff, to show the interval depends on the product.
    call = np.maximum(toy.X2_GRID[None, :] - 100, 0) * np.ones((3, 1))
    lo, hi, _, _ = mot_bounds(call)
    print(f"\nVanilla call (X2-100)^+: [{lo:.4f}, {hi:.4f}]  (a point: the surface fixes it)")

    print("\nWhy this baseline dies: LP unknowns with 20 grid prices per date")
    for T in (2, 3, 4, 6):
        print(f"  {T} dates: {20 ** T:>12,d} unknowns")


if __name__ == "__main__":
    main()
