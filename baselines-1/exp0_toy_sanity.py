"""
EXPERIMENT 0 -- Toy sanity checks (the debugging harness)
=========================================================
Question: do our basic measurement functions give the hand-checked numbers
from the lecture notes?  Nothing is learned here.  Every later experiment
relies on these functions, so this one must pass first.

Checks
  (a) call prices of the toy surface and Breeden-Litzenberger butterflies
      recover mu1 and mu2;
  (b) the surface is in convex order (same mean, calendar spreads >= 0),
      so a legitimate simulator exists (Theorem 1);
  (c) defect, marginal error and exotic price of the four named models.

Expected (notes, Ch. 3 and Day 2 appendix):
  model         defect   price of |X2-X1|   matches surface?
  tree          0        5                  yes
  min table     0        2.5                yes
  independent   5        9.375              yes
  non-example   1.25     6.25               no  (W1 on date 2 = 1.25)
"""
import numpy as np
from common import toy


def call_prices(mu, grid, strikes):
    """C(K) = E[(X - K)^+] under one date's distribution (zero rates)."""
    return np.array([(mu * np.maximum(grid - K, 0)).sum() for K in strikes])


def butterflies(C, dK):
    """Breeden-Litzenberger on a grid: Pr(X = K) = (C(K-dK) - 2C(K) + C(K+dK)) / dK."""
    return (C[:-2] - 2 * C[1:-1] + C[2:]) / dK


def main():
    strikes = np.arange(70, 131, 10.0)
    C1 = call_prices(toy.MU1, toy.X1_GRID, strikes)
    C2 = call_prices(toy.MU2, toy.X2_GRID, strikes)

    print("(a) Toy option surface")
    print("    strike   ", strikes)
    print("    C1(K)    ", C1)
    print("    C2(K)    ", C2)
    print("    butterflies date 1 (K=80..120):", butterflies(C1, 10.0))
    print("    butterflies date 2 (K=80..120):", butterflies(C2, 10.0))
    assert np.allclose(butterflies(C1, 10.0), [0, .25, .5, .25, 0])
    assert np.allclose(butterflies(C2, 10.0), toy.MU2)

    print("\n(b) Convex order test")
    m1, m2 = toy.MU1 @ toy.X1_GRID, toy.MU2 @ toy.X2_GRID
    print(f"    means: {m1}, {m2};  calendar spreads C2-C1: {C2 - C1}")
    assert m1 == m2 == toy.S0 and np.all(C2 - C1 >= 0)

    print("\n(c) The four named models")
    print(f"    {'model':<13}{'defect':>8}{'W1 d1':>8}{'W1 d2':>8}{'price':>9}")
    models = {"tree": toy.tree_table(), "min table": toy.min_table(),
              "independent": toy.independent_table(), "non-example": toy.nonexample_table()}
    for name, P in models.items():
        e1, e2 = toy.marginal_errors(P)
        print(f"    {name:<13}{toy.defect(P):8.3f}{e1:8.3f}{e2:8.3f}{toy.price(P):9.3f}")

    assert np.isclose(toy.defect(models["independent"]), 5.0)
    assert np.isclose(toy.defect(models["non-example"]), 1.25)
    assert np.isclose(toy.price(models["independent"]), 9.375)
    print("\nAll toy sanity checks passed.")


if __name__ == "__main__":
    main()
