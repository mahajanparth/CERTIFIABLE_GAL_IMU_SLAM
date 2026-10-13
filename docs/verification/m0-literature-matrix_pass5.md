# M0 adversarial verification, pass 5: literature matrix (narrow)

**Verifier:** independent pass 5. **Branch:** `m0-literature-matrix` at `d939780`. **Date:** 2026-10-13.
**Under test:** M0-C01..C15, M0-C19 (dispatch brief only), and the M0-F12 fix (`d939780`).

Status: IN PROGRESS. Early result recorded below; sections are extended in later commits.

## M0-C19: REFUTED (runtime, gtsam 4.3.0)

`(N1 * N2).velocity() == v1 + R1 v2`, not `v1 + v2`. 1000 random NavStates:
max |v - (v1 + v2)| = 65.3, max |v - (v1 + R1 v2)| = 7.1e-15. Example: N1 = NavState(Rz(90 deg), 0, [1,0,0]),
N2 = NavState(I, 0, [1,0,0]) gives velocity [1, 1, 0], not [2, 0, 0].

Suite on the clean tree at d939780: 143 passed.
