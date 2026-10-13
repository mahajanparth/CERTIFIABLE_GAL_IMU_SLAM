# Stage-B verification, pass 6: M2 baselines (`stage-b/m2-baselines` @ `b4e8420`)

Sixth independent adversarial pass. It follows `3ce49be` (WIP) and `3c28d06`, which say they
fix pass-5 finding M2-F13 (CayleyFactor.dRbar pinned by no test) by adding `test_c13d` and
replaying R01/R02/R04/R05/R10 in `tests/test_mutation.py`, and `b4e8420`, which edits the
M2-C22 statement to name the metric of each quoted rotation gap.

Scope: the F13 ruling; every claim in `docs/claims/M2.yaml` (C01-C25); one claim that exists
only in the pass brief, **M2-C29**; a field-by-field mutation sweep of the three baseline
constructors; a replay of earlier mutants as real file edits.

Rules kept: I did not read `docs/02_*` or any derivation narrative, any `seeds*.yaml`, or
anything under `.cgis_SEEDS_BACKUP*`. All scratch code and mutant copies lived in the session
scratchpad, outside the worktree. I did not edit `src/`, `tests/` or `docs/claims/`.

*(In progress -- this file is committed early and extended.)*

## 1. Independent references (no `src` imports)

Scratch module `ind/core.py`: constants retyped from the M2-C01 text; `Exp = expm o hat`,
`Log = logm`; `Cay = inv(I - phi^/2)(I + phi^/2)`; `J_Cay` typed from Eq. (4); the Eq. (9)
Euler sums with a pluggable retraction; the exact increment from ONE 5x5 `expm`; central-FD
bias Jacobians at eps = 1e-5; my own Eq. (19) typed as `R_j - A - 1/2 R_j xi^ - 1/2 A xi^`.
`ind/c06.py`: a Rodrigues-based noisy Euler integrator and a 200-step x 6-direction central FD
of it (eps = 1e-5) for the covariance. `ind/claims2.py`: my own 5x5 Gal(3) matrices, my own
ZOH step `V_k = expm(dt [[w^, a, 0],[0,0,1],[0,0,0]])`, my own FD Jacobians, and my own
factor graph (prior sigmas 5e-2 / 2e-1, GPS sigma 0.5, densities 3e-4 / 2e-2, random-walk
3e-5 / 5e-4 -- all different from the test's). `src` is imported only as the system under
test.
