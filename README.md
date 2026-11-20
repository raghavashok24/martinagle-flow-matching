# Martingale Flow Matching (MFM)

## 1. Project summary

Banks price complex options by simulating thousands of possible future price
paths. A simulator can only be trusted if it passes two tests:

1. **No free money.** At every point on every path, the average next price
   equals the current price, so no trading strategy profits inside the
   simulator.
2. **Market match.** On each option expiry date, the simulated prices have
   exactly the distribution that today's option prices imply.

Exact methods pass both tests but stop working beyond one or two assets or
dates. Neural generators scale, but leave some free money behind.

MFM alternates two corrections until both tests hold: a flow-matching step
that matches the market, and a step that removes free money, guided by a
neural "trader" that searches for profit. The two measured errors are then
turned into a certificate: a bound on how wrong the model's prices can be.

**Team:** Aarav Jamdar (mathematics), Raghav (finance and machine learning)

## 2. Repository

```
lecture-notes/
  master_lecture_notes.pdf   all notes in one document (start here)
  aarav-notes/               Aarav's daily handoffs
  raghav-notes/              Raghav's chapters
baselines-1/                 first baseline experiments (see its README)
```

## 3. Timeline

**Lecture notes** (seven-day build)

| Day | Output | Owner |
|---|---|---|
| 1 | Chapter 1; formula inventory; reference list | Raghav; Aarav |
| 2 | Chapter 2; toy examples | Raghav; Aarav |
| 3 | Chapter 3: mathematical foundations | Aarav |
| 4 | Chapter 4: generative modelling | Raghav |
| 5 | Chapters 5 and 6: the method and the v2.0 developments | both |
| 6 | Chapters 7, 8, 0 and 9 | both |
| 7 | Cold read, polish, freeze | both |

**Research** (twelve-week plan from the proposal; weeks are relative)

| Weeks | Work |
|---|---|
| 1–2 | Theory: pricing certificate in the discrete case; convergence write-up |
| 3–6 | Implement MFM; experiments E1 (synthetic truth) and E2 (exact solvers) |
| 7–10 | Experiments E3 (SPX), E4 (baskets), E5 (deep hedging) |
| 11–12 | Multi-asset scaling; paper assembly |
