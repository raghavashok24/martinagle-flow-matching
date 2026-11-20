# MFM — initial baseline experiments

First-round baselines for Martingale Flow Matching. Nothing here trains MFM
itself. The point is to fix the reference numbers and the evaluation code
that the MFM implementation will be debugged against (the proposal's
"exact-solver comparisons double as the debugging harness").

    pip install -r requirements.txt
    python run_all.py            # everything, about 30 seconds on a laptop CPU
    python exp1_mot_bounds_lp.py # or one experiment at a time

| File | Experiment | Proposal link | Answers |
|---|---|---|---|
| `exp0_toy_sanity.py` | 0. Toy sanity checks | harness | Do defect, W1, prices, Breeden–Litzenberger give the hand-checked toy numbers? |
| `exp1_mot_bounds_lp.py` | 1. Exact MOT price bounds (LP) | E2 | What interval of prices does no-arbitrage allow? Toy: [2.5, 5]. |
| `exp2_entropic_bridge_ipf.py` | 2. Entropic bridge by exact alternating projections | E2, Thm 2 | Which single model does the loop converge to? Toy: price 4.641, KL 0.602. |
| `exp3_defect_discriminator.py` | 3. Neural trader vs exact defect | Lemma 3, step 3a | Does the discriminator recover a known defect? |
| `exp4_heston_synthetic.py` | 4. Heston truth + naive generators | E1 | How do "matches surface, unfair" and "fair, misses surface" generators score? |

Shared code: `common/toy.py` (the two-date toy from the lecture notes and all
measurements on it) and `common/trader.py` (the bounded, predictable neural
trader used as the discriminator).

## Not in this round

Rough Bergomi, Dupire local volatility, the neural-SDE market model, SPX data
(E3), multi-asset baskets (E4), deep hedging (E5), ε-annealing toward the MOT
bounds, and MFM itself.

## Notes

- Experiment 2 prints two tables. The first does one row+column rescale per
  pass and reproduces Appendix A of the lecture notes digit for digit. The
  second does the exact marginal projection per pass. Same limit, different
  intermediate numbers.
- The proposal's drift-removal step (3b, Theorem 4) is deliberately not
  implemented: the lecture notes argue it is false as stated. The martingale
  step here is the exact row tilt.
- Zero interest rates and dividends throughout, European options only.
