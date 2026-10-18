# Stage B verification, pass 8: M3 QCQP derivation (branch stage-b/m3-qcqp-derivation)

Independent adversarial verifier, pass 8 (agent verify-M3-008). Base commit b4de3c3, the implementer's response to the
pass-7 findings M3-F11 (blocking) and M3-F12. Scope: src/qcqp/*, tests/test_qcqp.py, tests/test_mutation_m3.py,
docs/claims/M3.yaml (M3-C01..C35, C39), plus one claim given only in the brief, M3-C47. I did not read
docs/03_qcqp_derivation.md, experiments/m3_scratch/ or any seeds file. I did not edit src/, tests/ or docs/claims/.

Every Python run used `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` under a hard `timeout`. My scripts lived
in an untracked `_v8scratch/` inside the worktree (deleted before the commit). They were:

* `ref8.py`: a new independent reference. It imports neither src nor tests. It contains:
  * its own retyped literals, its own GTSAM PIMs (own params) and its own bias-order conversion;
  * Ubar as a 5x5 matrix from `pim.predict(NavState(), b)` and its own G (the CLAUDE.md corrected endpoint, built from
    `deltaTij()`);
  * J as a 200-sample least-squares fit of the **scipy `logm` chart** of `Ubar^-1 U(bbar + d)` (fit residual 1.0e-15 to
    1.7e-15);
  * the ground truth from a **Van Loan block exponential** with Rodrigues R_0;
  * its own encoder (column-major via `R.T.reshape`), its own block decoder and its own SVD projection.

  **Re-implemented core routines:** every constraint family as matrix equations on the blocks of an *arbitrary* z (in
  src's emission order, compared entry by entry); the C07 residual forms on arbitrary z; the cost on arbitrary z (iso /
  partial / aniso); and the cost of a state from 5x5 cleared Gal(3) residuals.
* `claimsA.py`, `claimsB.py` (own GTSAM evaluation: `GalileanImuFactor.unwhitenedError` on own PIMs weighted by own taus /
  W_R / G2 / W_vp, plus `PriorFactorPose3`, `PriorFactorVector`, `GPSFactor`, `BetweenFactorConstantBias` summed as
  2 graph.error), `c47.py`, `cmp8.py`, `prop8.py`, `hunt8.py`, `probe_fd.py`, `meas8.py`, `eff8.py`, `eq8.py`, `eq8b.py`.
* `dump8.py`, `cmpdump.py`: record **638 observable output groups** of a src tree and diff two trees. The groups cover every
  Instance / EdgeData field with its writeable chain (both bbar), the D / J_* properties, z at 6 states x 2 layouts x 2
  scales, every constraint family (meta, dense A, residuals on 7 z), flat / family_residuals / family_max, terms / Q /
  factored / dense / whitened for iso, partial, aniso (3 caller weights), an unknown variant and aniso-on-BASE, ~90
  decoder probes x 2 project values (scales 1e-320..1e305, threshold ties, NaN / inf, aniso rho, dtypes, shapes, random),
  38 `project_so3` inputs, BASE / ANISO / 4-KF-3-edge layouts (plain and aniso), vech, every forms helper, every boundary
  helper (19 gravity inputs, PIMs of 1 / 120 / 200 / 295 steps, increment at 3 biases, J at two steps, cov blocks), the
  fixture constants and functions, exports, State.copy, and the serialisation / FrozenDict-contract vectors.
  Two dumps of HEAD are identical (0 / 638 differ).
* `harness8.py`, `mutants8.py`, `replay8.py`: 150 new mutants plus 11 replays and a control. Each mutant is a real
  exactly-once textual edit (pattern count checked == 1) on a temp copy of src + tests/test_qcqp.py + conftest +
  pyproject + fixtures with `PYTHONPATH` stripped; the judge is the **full** tests/test_qcqp.py (slow tests included)
  with `-x`, 900 s timeout per mutant (none timed out; slowest 188 s). Survivors are dumped and diffed against HEAD.

## Verdict summary

* **Clean suite: 87/87 pass (70.6 s).** `tests/test_mutation_m3.py`: **109/109 pass (151.6 s)** (107 mutants + well-formedness
  + control).
* **M3-F11 (pass 7, blocking): FIXED.** The 5 F11 survivors and the 5 p9-fd mutants, replayed as real exactly-once edits
  and judged by the full suite, are all killed; the unmutated control passes (Sec. 1).
* **M3-F12: FIXED** for (a) and (b); (c) is documented on `Instance.pims` (accepted; GTSAM behaviour). Residual low
  items go to M3-F14.
* **No claim-level output changed.** 636 of 638 groups are bitwise identical between b0b3a27 and b4de3c3; the 2 that differ
  are exactly the intended F12 changes (Sec. 1). This corroborates the implementer's 2072/2072.
* **Claims: 36 CONFIRMED (C01-C35, C39). M3-C47 REFUTED**: det(I_5 - x^/2) = **1 + |omega_x|^2/4**, never singular. 0 UNTESTABLE.
* **Fresh sweep: 150 new mutants over all 8 files of src/qcqp, plus 11 replays.** 79 new + 11 replays are killed; 71 new
  mutants survive. **58 survivors are equivalent and 13 are NON-EQUIVALENT** (1 numerical, 12 low contract / freeze
  items).
* **GATE (zero surviving non-equivalent mutants): FAILED** (new finding M3-F13, Sec. 6). The numerical one is the pass-7
  Dt-literal defect class reappearing in `cost.py`: `Dt, g = edge.Dt, inst.g` -> `Dt, g = 0.5, inst.g` moves the C24 cost
  by **5.0e-13** (rule-1 bound 1e-13) and passes the whole suite.

## 1. Rulings on M3-F11 and M3-F12

### Replays (real exactly-once edits on b4de3c3, full suite, `-x`)

| mutant | result: first failing test |
|---|---|
| control (unmutated) | passes, 55 s; dump identical to HEAD (0 / 638) |
| p7v-gb-inc-Dt-literal | killed by test_instance_matches_independent_gtsam_reference[true] |
| p7v-in-fd-reduce-dict | killed by test_frozendict_standalone_round_trip_and_no_mutable_attribute |
| p7v-in-fd-init-args0 | killed by test_instance_arrays_have_no_writable_base |
| p7v-in-fd-init-alias (pass-7 form `dict(...)` -> `args[0]`) | killed by test_instance_arrays_have_no_writable_base |
| p7v-in-fd-len | killed by test_frozendict_mapping_contract |
| p7v-in-fd-getitem-get | killed by test_frozendict_mapping_contract |
| p9-fd-d-list | killed by test_frozendict_standalone_round_trip_and_no_mutable_attribute |
| p9-fd-ro-off, p9-fd-ro-nowrite, p9-fd-ro-nobase, p9-fd-ro-nocopy | killed by test_frozendict_mapping_contract (4/4) |

**11/11 killed. M3-F11 FIXED.**

### Robustness of the tightened dv / dp pins and the new Dt test

* Clean, src vs `deltaXij` (my own PIMs): dv **0**, dp **4.44e-16** on every edge, both bbar. Headroom: dv infinite, dp
  3.4x. The pin compares two GTSAM computations in one process, so it does not depend on the platform's BLAS.
* The Dt-literal mutant, all 4 failing tests listed without `-x`: test_instance_matches[true] and [zero] fail on **dv
  3.55e-15 > 1e-15** (3.5x margin); **dp moves to 1.554e-15 vs the 1.5e-15 pin, a 3.6 % margin**, so the dp pin alone is
  marginal. Both `test_increment_uses_the_pim_duration[120]` and `[295]` also fail, at `U[3,4] == Dt` (0.5 vs
  0.3000000000000002 / 0.7374999999999953) and by O(1) in v / p.
* Sub-ulp variants of the same defect are also killed: `Dt = deltaTij() * (1 + 2.3e-16)` (killed by the [120] test),
  `round(deltaTij(), 12)` and `float(np.float32(deltaTij()))` (killed by the instance test).
* **Judgement:** robust. The non-0.5 Dt test is the load-bearing kill (an O(1) signal). The dp pin carries almost no
  margin for this defect, but it is not needed. Remark: the same defect class in `cost.py` is **not** covered (M3-F13 a).

### M3-F12: FrozenDict redesign hunt (`probe_fd.py`)

| check | b4de3c3 |
|---|---|
| `_d` | a `tuple` of `(key, value)` 2-tuples; `fd._d = ...`, `fd.x = ...` and `fd.__class__ = dict` raise AttributeError; no `__dict__` |
| values on the Instance path | read-only, `base is None` (also after copy, deepcopy and pickle protocols **0-5** of a standalone `ed.cov`; deepcopy / asdict / astuple of the Instance give FrozenDict with read-only arrays) |
| `len`, `in`, `get`, `get(k, d)`, missing key, unhashable key | as dict: 2 / 9; KeyError; `0 in gnss` False; `get(0)` None; unhashable raises TypeError |
| equality | `FD == dict`, `FD == FD` (order-insensitive), `FD == mappingproxy` all True; `!=` correct |
| hashing | unhashable (TypeError), as dict |
| iteration / keys / values / items | insertion order, identical to the source dict (incl. the 9 cov keys); KeysView / ValuesView |
| constructor | dict semantics: kwargs, pair iterables, duplicate keys (last wins), empty; caller's dict never aliased |
| caller arrays | writable or view values are copied (caller's array stays writable and unshared); **a caller's read-only owning array is kept by reference** (documented, tested) |
| mutation paths that remain | (i) `fd[k].setflags(write=True)` succeeds, because the stored arrays are owning arrays, and then writes reach the cached instance. This is the same deliberate-bypass class accepted in pass 6 for every Instance array. (ii) `fd.__init__({...})` re-binds `_d` (dunder). (iii) A caller who kept a read-only owning array can re-enable writes on it and change the FD. (iv) Non-ndarray mutable values (lists, object-array cells, a masked array's `.mask`) are not frozen. (i)-(iv) are all outside src's own use |
| contract deviation from the MappingProxyType it replaced | `reversed(fd)` raises TypeError; `fd.copy()` raises AttributeError; `fd \| {}` raises TypeError. mappingproxy supports all three. No src or test consumer uses them (M3-F14) |

**Ruling M3-F12:** (a) **FIXED**: no attribute exposes a mutable container. (b) **FIXED**: standalone round trips stay
FrozenDict with read-only, base-free arrays for every protocol. (c) **Accepted as documented** on `Instance.pims` (GTSAM
pybind11 behaviour; protocol >= 2 is tested). The residual items above are low and not gate-blocking (M3-F14).

### No claim-level output changed (b0b3a27 vs b4de3c3, 638 groups)

636 groups are bitwise identical. The 2 that differ:

* `fdcontract`: the stored value no longer shares memory with the caller's array and is read-only; `_d` is a tuple.
* `ser`: a FrozenDict pickled (protocols 2 / 4 / 5) or deep-copied on its own, and `asdict(I)["gnss"]`, now come back
  read-only and base-free.

Every instance value, z, constraint matrix, residual, cost, Q, whitened residual, decoder output, projection, boundary
helper, fixture value and export is unchanged.

### The new tests and their bindings

* `test_increment_uses_the_pim_duration[120/295]`, `test_frozendict_mapping_contract`,
  `test_frozendict_standalone_round_trip_and_no_mutable_attribute` and `test_instance_pickles_with_every_protocol_from_2`
  are real, and are unbound with an explicit `NOT claim-bound` reason. **Honest.**
* The conftest thread pin is harmless (it sets env vars before numpy import; the mutation harness inherits it).

## 2. Independent re-checks (the reference imports no src)

| check | result |
|---|---|
| instance vs ref8 PIMs (both bbar, both edges) | dR, dv, dp **0**; taus, W_R, G2, W_vp **0** (relative); Dt 0.5000000000000003; truth (Van Loan) 3.3e-15; gnss 3.3e-15; rw, g, b_true 0 |
| J: src (h = 0.1) vs ref8 (logm LS fit) | 6.2e-15 to 1.0e-14 |
| z_of_state vs own encoder, 30 random states x 2 layouts | <= 2.6e-15 (true) / 4.4e-15 (zero) relative |
| **families: src A_k vs ref8 matrix equations on ARBITRARY z (20 z ~ N(0, I)), base and aniso, both bbar** | same family keys and order, same count and order within every family. Max relative difference: P0 0; P1 / P2 / RA1 <= 3.6e-16; RB2 / RS / P4 / RR <= 1.3e-15; P3 / RB1 / RC1 / RG / RL / RH <= 2.7e-14 (J-driven) |
| cost on ARBITRARY z vs own (iso / partial / aniso, both bbar, 50 z each) | dense <= 6.4e-16 relative, factored <= 2.3e-16 relative |
| truth cost (bbar = b_true) | factored (own truth) 6.17e-22, src truth_state 6.71e-22, own 5x5 6.23e-22; **dense z^T Q z 1.2e-8** (cancellation; R1) |
| truth cost (bbar = 0) | factored 3.2967763859e-3, own 5x5 3.2967763858e-3 (surrogate gap, R9) |

### Property tests (1e4 samples per configuration; own encoder; src constraint matrices and costs)

Sampler: R = Rodrigues(2 N(0, I3)), v, p ~ N(0, I), b ~ 0.3 N(0, I6).

| instance | family max on lifted states | C24: cost vs own 5x5, / max(1, J) (dense / factored) |
|---|---|---|
| true, base | P0 0, P1 1.9e-15, P2 1.7e-15, P3 1.0e-14, RA1 1.9e-15, RB1 6.7e-16, RB2 1.5e-15, RC1 8.3e-15, RG 5.6e-15, RL 9.2e-15, RH 9.2e-15, RS 1.3e-15 | 9.3e-16 / 4.3e-16 |
| true, aniso | as base (<= 1.1e-14); P4 2.3e-14, RR 3.1e-13 | iso 6.6e-16 / 3.5e-16; aniso 8.3e-16 / 4.3e-16 |
| zero, base | <= 1.7e-14 | 1.1e-15 / 4.1e-16 |
| zero, aniso | <= 1.9e-14; P4 2.2e-14, RR 3.4e-13 | iso 6.5e-16 / 4.3e-16; aniso 7.9e-16 / 4.7e-16 |

These are within C09-C12 (1e-12 base; 1e-10 aniso) with >= 53x headroom, and within C24 (1e-12) with >= 900x headroom.

**Decoder vs own decoder, 1e4 inputs x 2 project values:**

* The inputs alternate base / aniso, use scales {1, -1, -3.7, 0.05, 1e-6, 1e6}, add noise on every 3rd, put s at
  1e-12 max|z| (+-1e-9) on every 11th, a 1e14 entry on every 17th, float32 on every 19th, and NaN / +-inf on every 23rd.
* There are **0 raise / no-raise disagreements**; 2682 inputs raised in both decoders.
* The decoded blocks are bitwise equal (max difference 0). The projected rotations are orthogonal to 2.7e-15.

### Spurious-feasible-point hunt

TRF (`xtol = ftol = gtol = 1e-15`, analytic Jacobian 2 A_k z, src's A_k); lifted-ness judged by ref8's decoder and encoder,
consistency = max|z - s z(state(z))|.

| family set | converged | consistency | det R_k |
|---|---|---|---|
| primary P0-P3, bbar = 0 (rng 7) | 200/200 | 2.6e-13 | 600 / 600 +1 |
| primary, bbar = b_true | 60/60 | 7.4e-14 | 180 / 180 +1 |
| aniso primary P0-P4 | 60/60 | 1.7e-13 | 180 / 180 +1 |
| all 12 families, bbar = 0 | 40/40 | 6.1e-15 | 120 / 120 +1 |
| drop P2 | 60/60 | 3.8e-14 | **96 / 180 det -1** |
| drop P2, keep all 8 redundant families | 40/40 | 5.2e-15 | **34 / 120 det -1** (confirms pass-7 R2) |
| drop P3 | 60/60 | **3.6e2** | +1 (liftings off their definitions) |
| drop P1 | 60/60 | 1.9e1 | min \|det\| **5.7e-73** |

No spurious point exists for any set that contains P0-P3.

## 3. Claims

Each verdict rests on my own computation, with src as the object under test.

| claim | verdict | evidence (this pass) |
|---|---|---|
| C01 | CONFIRMED | n 124 / 136; s 123, z[123] = 1; rho_v(0) = [124,127), rho_p(1) = [133,136); src z vs own encoder 3.3e-15 at truth, <= 4.4e-15 relative over 60 random states x 2 layouts |
| C02 | CONFIRMED | 112, 12; the V' columns of imu{0,1}.vel are exactly -1/2 I3 |
| C03 | CONFIRMED | all 7 triples satisfy n = 39K + 6M + 16; offset vs the direct count 39 for each |
| C04 | CONFIRMED | own FD (h = 1e-6, Gal3.Logmap of own 5x5 U) on 4 fixture cases + 6 random time-varying streams (150 samples) with random bbar: J[0:3,3:6] = 0 and J[9] = 0 exactly; \|J_nu,a - J_R\| <= 3.95e-10; \|J_R\| 0.37-0.50. Own series sum (-ad_x)^n/(n+1)! (own ad, checked against the matrix commutator) vs FD of Gal3.Expmap, 200 x at \|x\| = 0.8: 1.95e-10; block differences exactly 0. Shipped J: zeros exact, \|J_nu,a - J_R\| <= 7.6e-15 |
| C05 | CONFIRMED | own algebra, 2000 draws per bbar: max\|r\| 1.8e-15; \|t_j\| 0; bottom rows exactly 0 for arbitrary X_j; arbitrary X_j min max\|r\| 0.95 / 1.09; A34 = B34 = 0.5000000000000003 |
| C06 | CONFIRMED | 2000 draws per bbar, \|eps\| = 0.3, eps_alpha = 0, scipy expm: 2.5e-15 / 3.1e-15 |
| C07 | CONFIRMED | src forms on z(state) vs own 5x5, 500 states per bbar: 5.3e-15 / 8.0e-15 (<= 1e-12) |
| C08 | CONFIRMED | coefficient blocks exact (deviation 0, both bbar), bias columns exactly 0; truth 6.7e-22; perturbed Y_0 min over 20: 3.4e-5 (>= 1e-6) |
| C09-C12 | CONFIRMED | truth and property run (Sec. 2): base <= 1.9e-14 (<= 1e-12); P4 2.3e-14, RR 3.4e-13 (<= 1e-10) |
| C13 | CONFIRMED | own z_truth + 1e-3 N, min over 20 of each family max: smallest 1.22e-5 (RB1, base), 1.33e-5 (RB1, aniso); > 1e-7 |
| C14 | CONFIRMED | sizes exactly as stated; every family full rank (own vech, tol 1e-9); primary 106 / 106 (aniso 118 / 118); joint 472 -> 453 (aniso 490 -> 471) |
| C15 | CONFIRMED | own row-handedness matrices built entrywise from eps: each equals a src P2 matrix, mismatch 0 (all 3 KF) |
| C16 | CONFIRMED | rank 11 for each KF; trace identity exactly 0 |
| C17 | CONFIRMED | every A_k exactly symmetric; homogeneity <= 6.5e-17; exactly one b != 0, b = 1 (base and aniso) |
| C18 | CONFIRMED | 200/200 converged, consistency 2.6e-13, 600 / 600 det +1. Drop P2: 96 / 180 det -1. Drop P3: 3.6e2. Drop P1: \|det\| 5.7e-73 |
| C19 | CONFIRMED | own 9 equations, 300 starts: 237 SO(3), 63 zero, 0 other |
| C20 | CONFIRMED | violated: P2 2.0, RH 8.5e-2, RS 1.5e-2; satisfied <= 3.5e-16 (both bbar) |
| C21 | CONFIRMED | P1 = RA1 = 1.0 (both bbar); all others <= 3.9e-16 |
| C22 | CONFIRMED | rotated: P3 1.5e-2, RL 1.4e-2, RH 1.5e-2, RS 3.9e-3; negated: P3 9.6e-2, RL 8.4e-2, RH 9.0e-2, RS 1.3e-2; satisfied <= 3.3e-16 |
| C23 | CONFIRMED | residuals of -z equal those of z exactly (base / aniso, both bbar); cost difference 0.0 (dense and factored, every variant); decode(-z) == decode(z) exactly, both project values |
| C24 | CONFIRMED | 1e4 lifted states per (bbar, layout) vs own 5x5 cost: dense <= 1.1e-15, factored <= 4.7e-16 (<= 1e-12). Truth state, bbar = 0: iso 1.09e-13, aniso 3.4e-15 (R7) |
| C25 | CONFIRMED | iso / partial / aniso at both bbar: Q exactly symmetric; min/max eig >= -2.3e-16; Q[b, non-b] = 0 exactly; bias block equals own rw matrix exactly |
| C26 | CONFIRMED | truth factored 6.17e-22 (<= 1e-15); own GTSAM evaluation, 50 states per eps: 1.66e-4, 3.72e-6, 2.82e-7 (bounds 5e-4, 5e-5, 5e-6) |
| C27 | CONFIRMED | own PIMs / GalileanImuFactor.unwhitenedError vs src forms at truth. lam = 1: rot / vel / pos +2.011e-2 / -1.223e-2 / -2.098e-3 (<= 3e-2). Halving factors: rot 0.250 / 0.249 / 0.249; vel 0.501 / 0.500 / 0.500; pos 0.504 / 0.502 / 0.501 |
| C28 | CONFIRMED | own Cayley vs scipy expm on own Ubar: 2.0880e-7, 2.6048e-8, 3.2527e-9, 4.0639e-10; factors 8.02 / 8.01 / 8.00 |
| C29 | CONFIRMED | half 2.3e-7 (<= 5e-6); literal Rai weight 0.83 (>= 0.1) |
| C30 | CONFIRMED | src whitened residual, own local parametrisation (Rodrigues), h = 1e-6: rank 45 / 45 both bbar; cond 3.946e7 / 4.527e7 |
| C31 | CONFIRMED | own encoder, 200 random R_0: 3I constant 4.020000 (ptp 2.1e-13); diag(1,2,5) range [1.371, 6.692] (ptp 5.32 > 1.0) |
| C32 | CONFIRMED | own sampler (rng 2), own encoder, numpy lstsq on 2278 monomials: 2.4e-15, 2.7e-15 (<= 1e-12); diag(1,2,5) 7.9e-2; random SPD 1.1e-1; diag(1,1,1.01) 8.8e-4; rank 2015 |
| C33 | CONFIRMED | partial vs own GTSAM evaluation: 1.29e-4, 3.09e-6, 2.34e-7 (bounds 0.05 eps); D eigenvalues >= 3.4602e7 (src = own bitwise); G2 correlation -0.867 |
| C34 | CONFIRMED | body 3.8e-3, 1.8e-4, 2.5e-5 (<= eps); world 8.2e-2, 4.2e-2, 7.5e-2 (>= 1e-2) |
| C35 | CONFIRMED | rng 4, own encoder: rank 2015, null 263, gap 1.7e-7 -> 4.5e-16. Spans 60 / 92 / 106 / 136 / 226 / 236 (P0 excluded: it is inhomogeneous). Max projection residual 2.9e-11 (<= 1e-10). R5 still applies |
| C39 | CONFIRMED | Dt 0.5000000000000003 for both edges; bbar = 0 taus 6.9204e7 / 4.8049e5 / 5.8906e6 and 6.9204e7 / 4.8167e5 / 5.8972e6 (own = src); rw diagonal 5.5402e9 / 2.2222e5 |
| **C47** | **REFUTED** | below |

**M3-C47 (brief only): REFUTED.** x^ = [[phi_x, u, w], [0, 0, alpha], [0, 0, 0]], so I_5 - x^/2 is block upper triangular with
diagonal blocks (I_3 - phi_x/2, 1, 1). Hence det(I_5 - x^/2) = det(I_3 - [omega]_x/2) = **1 + |omega_x|^2/4 >= 1**, with a
plus sign.

* Over 1e4 random x in gal(3) (|x| from 1e-3 to 1e2):
  * the claimed 1 - |omega|^2/4 misses by up to **2.0 (relative)**;
  * 1 + |omega|^2/4 holds to 7.2e-15 relative;
  * the minimum determinant is 1.0000000048.
* **Exact (mpmath, 50 digits), at |omega| = 2:** det = 2.0000000000000000 (to the float rounding of the input). The claim
  predicts 0.
* The smallest singular value of the map A -> A(I - x^/2) at |omega| = 2 is 0.73, so it is not singular and adds no
  spurious zeros: r = 0 is equivalent to the Cayley relation for every x.
* This is what C05's own source note and pre-flight F8 state ("1 + |omega_x|^2/4 > 0"). The singularity statement and the
  spurious-zero statement are false.

## 4. Mutation sweep

### 4.1 New mutants (150; ids `p8v-*`)

By file: instance 42, gtsam_boundary 26, constraints 22, lifting 20, cost 19, forms 10, fixture 9, `__init__` 2. Emphasis:

* the new FrozenDict / `_ro` / `__reduce__` code (22 mutants);
* Dt and time handling at every site (`increment` round / float32 / 1-ulp nudge, `cost.py`'s Dt, `dtb`);
* generic `Layout` arithmetic, API contracts (`flat`, `family_residuals`, sparse class, `project` truthiness);
* freezing and aliasing of module constants, rounding-order variants and dataclass frozenness.

**Killed (79), grouped by first failing test:**

* test_instance_matches_independent_gtsam_reference[true] (13): in-Svp-swap, gb-int-dropone, gb-int-dt2, gb-gl-tpos,
  gb-inc-tDt, gb-inc-round12, gb-inc-f32, gb-inc-swap, gb-cov-blockT, fx-m2-sign, fx-truth-vT, fx-BA-digit, fx-sint.
* test_c09 (9): cn-Rj-i, cn-P0-b0, cn-cyc-bad, cn-sides-swap, cn-bf-j, fm-cvec-T, fm-eps-noneg, fm-cross-sign, fm-linT-noop.
* test_frozendict_mapping_contract (8): fd-init-kwdrop, fd-getitem-next, fd-init-noro, fd-iter-sorted, fd-setattr-pass,
  fd-setattr-type, ro-and, ro-noflag.
* test_c01_index_map[true] (8): lf-dec-s-abs, lf-vp-off, lf-z-sfirst, lf-Yp-Rj-i, lf-W-T, co-M-dRT, co-rv-cvM-only,
  fm-cmat-T.
* test_make_params_rejects_non_z_up_gravity (4): g-le0, g-isnan, g-size3, g-x0only.
* Three each:
  * test_instance_pickle_and_deepcopy_round_trip: fd-iter-rev, fz-tuple-list, inst-unfrozen;
  * test_instance_arrays_have_no_writable_base: fz-viewcopy-off, rebuild-nofreeze-edge, gb-cov-T;
  * test_c04 instance J: J-fwd, J-h001, J-Ubarinv-swap;
  * test_c11: RG-noWT, RS-negw, RH-ea-neg.
* Two each:
  * test_frozendict_standalone_round_trip: fd-slots-weakref, ro-baseRO;
  * s-guard tie: SFLOOR-11, guard-lt;
  * exports: init-drop-truth, init-drop-build.
* One each:
  * bbar-alias: in-btrue-nocopy;
  * test_c25[iso]: in-dtb-pim;
  * **test_increment_uses_the_pim_duration[120]: gb-inc-Dtnudge**;
  * bias converters: gb-from-order;
  * floor: guard-floor290;
  * det correction: proj-det-noSign;
  * test_c18: gb-J-h02;
  * non-finite: bad-nan;
  * aniso rho guard: zmax-base;
  * State.copy: copy-asarray;
  * truth_state: ts-shared;
  * test_c14: RC1-dropP;
  * vech: vech-tril;
  * test_c12: P4-Rj;
  * whitening: wh-abs;
  * default variant: v0-half;
  * test_c24[aniso]: rho-swap;
  * test_c08: vecrows-rowmajor;
  * forms: sym-noop.

### 4.2 Replays

* F11 survivors (5), the pass-7 original alias form (1) and the p9-fd mutants (5): 11/11 killed (Sec. 1).
* `tests/test_mutation_m3.py`: **109 passed** (151.6 s).

### 4.3 Complete survivor list (71) with rulings

Measured effect = the 638-group dump diff against clean HEAD, plus targeted probes:

* `meas8.py`: error of the src cost vs the own 5x5 cost, / max(1, J), at the truth state and on 200 random states, per
  bbar and variant;
* `eff8.py`: |cost_mutant - cost_clean| / max(1, J) on the truth state and 100 random states, per bbar and variant;
* `eq8.py` / `eq8b.py`: error vs exact rational / mpmath references.

The rule-1 bound for the cost is 1/10 of C24's 1e-12, i.e. 1e-13. As in passes 3-7, a change in a claim-free contract (an
output of a src function on a valid input, or a side effect on a caller's or module's data) is non-equivalent unless a
proof shows no such output changes.

**Non-equivalent (13):**

| # | mutant | measured effect | class |
|---|---|---|---|
| 1 | **p8v-co-Dt-literal** (`cost.py`: `Dt, g = edge.Dt, inst.g` -> `Dt, g = 0.5, inst.g`) | 62 groups; the r_v / r_p constant shifts by 3.55e-15. Cost effect, bbar = 0 truth: **iso 5.0e-13**, aniso 2.8e-13, partial 2.7e-13. Error vs the own 5x5 at the bbar = 0 truth: iso **6.1e-13** (clean 1.1e-13), aniso 2.8e-13 (clean 3.4e-15), synthetic 1e6 aniso **3.05e-12** (clean 6.7e-13) | **medium-low, numerical.** Rule 1 fails (5e-13 > 1e-13); rules 2 and 4 fail (the excess is >> 2 ulp). The same defect class as F11 #1, at the one remaining site: the forms take Dt from a second place, and C07 (1e-12) and C24's 500-state test (relative to max(1, J), random states with J >> 1) are too loose to see it |
| 2 | p8v-lf-n-12 (`Layout.n`: `6 * n_edges` -> `12`) | `Layout(4, 3, aniso=True).n` = 187 instead of 193 (rho_v(2), rho_p(2) run past n) | low: generic Layout (K != 3) untested |
| 3 | p8v-lf-bias0-45 (`bias0`: `15 * n_kf` -> `45`) | `Layout(4, 3)`: biases overlap KF3's blocks; 10 layout groups | low: same |
| 4 | p8v-lf-o-18 (`_o`: `6 * n_kf` -> `18`) | `Layout(4, 3)`: Y(0) = [78, 87) instead of [84, 93), s = 168 instead of 174 | low: same |
| 5 | p8v-cn-flat-empty (`if names is None` -> `if not names`) | `flat(fam, [])` returns all 472 constraints instead of none | low: API contract |
| 6 | p8v-cn-fres-list | `family_residuals` values become lists (annotated `np.ndarray`); 4 groups | low: API contract |
| 7 | p8v-cn-csr-array (`sp.csr_matrix` -> `sp.csr_array`) | the class changes in 52 groups. `c.A * z` becomes an elementwise sparse product instead of the mat-vec (measured), against the annotation `A: sp.csr_matrix`. Pass-7's csc mutant was equivalent because csc_matrix keeps the `*` semantics; csr_array does not | low |
| 8 | p8v-lf-proj-istrue (`if project:` -> `if project is True:`) | `state_of_z(z, project=np.True_)` (or 1) silently returns unprojected rotations (measured det != 1) | low: a numpy-bool flag from M4 code would skip rounding |
| 9 | p8v-in-ro-exacttype (`isinstance(v, np.ndarray)` -> `type(v) is np.ndarray` in `_ro`) | a writable ndarray subclass (np.matrix, memmap) is stored aliased and writable in a standalone FrozenDict, against the docstring "every numpy value that is writable or a view is replaced by a read-only copy" | low |
| 10 | p8v-in-edge-unfrozen (`EdgeData` not frozen) | `instance("true").edges[0].tau_R = 0.0` succeeds (clean: FrozenInstanceError) and poisons the lru-cached instance and every later `terms()` | low: the F09 cache-poisoning class |
| 11 | p8v-in-g-nocopy (`g = F.G_W.copy()` -> `F.G_W`) | building an instance freezes the module constant `fixture.G_W` in place, and `I.g is fixture.G_W` | low: the pass-6 bbar-nocopy class, applied to a module constant |
| 12 | p8v-fx-v0-copy (`true_states` returns `V_0`, `P_0` themselves) | `true_states()[0][1] is V_0`; the instance build then freezes `fixture.V_0` / `P_0` in place (dump: writeable False) | low: same class |
| 13 | p8v-fx-streams-view (`np.tile` -> `np.broadcast_to`) | `streams(e)` returns read-only stride-0 views (all 200 rows share memory) instead of writable arrays; values equal | low: output contract of a public fixture function |

**Equivalent (58):**

| mutants | ruling and justification |
|---|---|
| fd-getitem-rev, fd-iter-dict, fd-len-dict, fd-reduce-dictarg, fd-reduce-type | proof: `_d` is built from a dict, so its keys are unique and in insertion order; `dict(reversed(...))`, `iter(dict(...))` and `len(dict(...))` agree with the tuple; `FrozenDict.__init__` accepts a dict or a pair tuple identically; no subclass of FrozenDict exists. 0 groups |
| fd-repr-raw | the repr text only (`FrozenDict(((1, 1.0),))`). No documented repr format and no consumer (cf. the pass-7 dict-repr change) |
| ro-copyF (F-order copy in `_ro`) | proof: the copy is value-exact, so the values are bitwise equal. Only contiguity of standalone-FD values differs; no documented layout. Unreachable on the Instance path (values arrive read-only and owning). 0 groups |
| fz-mapping-dict (`isinstance(x, Mapping)` -> `dict` in `_freeze`) | proof: build passes plain dicts; on the rebuild path a FrozenDict arrives already rebuilt read-only by its own `__reduce__`. 0 groups (incl. pickle / deepcopy vectors) |
| rebuild-inst-generic | proof as pass-7 #17: the edges are already rebuilt and frozen; `_freeze` leaves the pims unchanged. 0 groups |
| in-D-div2, gb-J-div, fm-sym-div, co-prior-R-sq | proof: multiplication / division by 2 and `**1` are exact (no under / overflow on the domain). 0 groups |
| in-tau-recip | tau_v / tau_p change by 1 ulp. Cost effect <= 4e-16 relative. Rule 1 (C39 4 significant figures; C24 1e-13) |
| in-S2-mul3, in-Wvp-pinv, co-Q-assoc, co-fc-assoc, co-partial-G2T | G2 / W_vp / Q / factored cost move at <= 1.3e-15 relative; partial / aniso cost error vs own <= 7.7e-16. Rule 1 (C33 >= 5e-6; C24 1e-13) |
| in-Dt-nofloat, in-WR-solve, gb-gal3-tnofloat, gb-to-nodtype, gb-g-abs, gb-int-sq, co-rv-gDt, co-gnss-w | 0 groups. Proofs: deltaTij is already a Python float; pybind converts t and int bias arrays exactly; abs(g_z) = -g_z on the guarded domain g_z < 0; `x**2` and `x*x` are both one correctly rounded operation; s has 0 / 1 entries, so the outer product commutes with the Dt scaling; `I / x` and `I * (1 / x)` give the same entry `1.0 / x`. The W_R solve is bitwise identical on both instances (the whole domain) |
| gb-gl-half (`-0.5 g Dt^2` -> `-(g Dt Dt) / 2`) | dp moves by 1 ulp (2.2e-16; 2 ulp of \|dp\| = 5.3e-16, rule 4 on dp). Cost effect <= 9.8e-14 (zero / aniso truth), within the 1e-13 rule-1 bound (narrowly). Error vs the own 5x5 is better than clean on iso and synthetic. Rule 1 |
| gb-J-h005 (FD step 0.05) | J error vs logm fit 1.65e-14 (clean 1.0e-14); C04 bound 1e-8. Cost effect <= 7.3e-14. Rule 1 |
| co-rp-DtDt (`Dt**2` -> `Dt * Dt`) | cost effect <= 3.8e-14. Rule 1 |
| lf-aniso-solve (`M.T @` -> `solve(M, .)`) | rho entries at ulp level; cost effect <= 2.8e-16. Rule 1 |
| lf-dec-recip (`v / s` -> `v * (1 / s)`) | 3000 random z, exact rational reference: clean 0 ulp, mutant <= 1 ulp of max\|v\|; excess 0.50 x (2 eps max\|value\|). Rule 4 |
| fx-exp-thr (1e-8 -> 1e-6), fx-exp-thr0 (`== 0.0`) | sup error vs mpmath expm over 2000 \|theta\| in [1e-12, 3e-6] is 1.1e-16 for clean and both mutants. Rule 2 |
| lf-proj-detprod | proof: U and Vt are orthogonal, so both expressions are +-1 to 1e-15 with the same sign. 0 groups |
| cn-mk-nosym, cn-resid-order, cn-RR-swap, cn-RB2-wwT, cn-P1-ss-Q, fm-matmul-T, fm-dot-mn | proof: every quadratic form is already exactly symmetric (pass-6 #5); sym(X) = sym(X^T) bitwise (commutative addition); `ss` and the e_s e_s^T matrix are identical. 0 groups |
| cn-order, cn-name-tuple | the same set of forms in a different family order / with differently formatted names. Equivalent by the pass-7 #6 (P2-rows) precedent |
| cn-fmax-nofloat | np.float64 is a float subclass with the same value. 0 groups |
| fm-hat-ba | the sign of zero entries only (-0.0 -> 0.0); cf. pass-7 #14 |
| in-gnss-order (`(1, 2)` -> `(2, 1)`) | iteration order of `I.gnss` only. The mapping is equal (Mapping `==`), and the only consumer sorts (`terms`) |
| co-gnss-unsorted, co-rw-edges, co-Gam-fullJ | proof on the domain: gnss is built in key order; n_kf - 1 == len(edges) for every built instance; J[0:3, 3:6] is exactly 0, so the added terms are exact zeros. 0 groups |
| co-Wvp-noconv, in-bbar-int, in-cov-dict, in-frozen-pims | every consumer promotes (precedent pass-7 #18); `np.array` copies; `dict(cov)` is re-wrapped by `_freeze`; the pims tuple is rebuilt with the same objects. 0 groups |
| co-term-unfrozen | Terms are freshly built per call and never cached, and `t.F` is already a writable fresh array, so attribute assignment adds no reach into shared state. 0 groups |
| in-prior-alias, in-lru-1 | the shared arrays are read-only owning arrays with identical values; `build_instance` is deterministic, so eviction rebuilds identical data. 0 groups |
| fx-hat-nofloat | the literal 0.0 entries force a float64 array for any numeric input. 0 groups |

Totals: 71 survivors. **Non-equivalent 13, equivalent 58.**

## 5. Tests that cannot fail, and untested surface (at b4de3c3)

1. Unchanged from pass 7: the same-LAPACK projection pin; the tie-precondition arithmetic; the C02 / C03 arithmetic; the
   test-side-only parts of C04, C15, C26, C33, C39; the MATH CHECK labels on C05 / C06 / C28; the scale invariance of
   C30's rank.
2. **The time constant used by the residual forms is not pinned to `EdgeData.Dt`.** C07 compares at 1e-12 absolute, and
   C24 at 1e-12 relative to max(1, J) on random states where J >> 1. Neither sees a 3.6e-15 shift of the gravity term.
3. **Generic `Layout(n_kf, n_edges)` with K != 3 is untested** (3 survivors). M4 will need longer chains.
4. API edges: `flat(fam, [])`, the `family_residuals` value type, the `Constraint.A` class and the `project` truthiness.
5. Freeze side effects on **module constants** (`fixture.G_W`, `V_0`, `P_0`), the EdgeData frozenness and the `streams`
   output flags. The bbar-alias test covers only the caller's bbar and `BIAS_TRUE`.

## 6. Findings and risks for M4

* **M3-F13 (new, gate-blocking): 13 non-equivalent survivors (Sec. 4.3 #1-#13).** One is numerical and 12 are low.
  Tests that would kill them:
  * (a) #1: evaluate `edge_residual_forms` on `z_of_state(truth)` at bbar = 0 and compare r_v, r_p with an own 5x5
    residual built with G(`pim.deltaTij()`) to <= 1e-15 absolute (clean: 0 / 4.4e-16-level). Alternatively assert the
    constant (s) column of r_v equals `-g * edge.Dt` bitwise.
  * (b) #2-#4: assert `Layout(4, 3)` and `Layout(4, 3, True)` index maps against an own formula (bias0 = 60, Y(0) =
    [84, 93), s = 174, n = 175 / 193); ideally build z on a 4-KF state.
  * (c) #5-#8:
    * `CN.flat(fam, []) == []`;
    * `type(v) is np.ndarray` for `family_residuals`;
    * `isinstance(c.A, sp.csr_matrix)` (as pass 7 remarked);
    * `state_of_z(z, project=np.True_)` projects.
  * (d) #9-#13:
    * a FrozenDict built from a writable `np.matrix` holds a read-only unshared copy;
    * `with pytest.raises(FrozenInstanceError): I.edges[0].tau_R = 0`;
    * after `build_instance`, `fixture.G_W`, `V_0`, `P_0` are still writable and not shared with the instance;
    * `streams(e)` arrays are writable and do not share memory across rows.
* **M3-F14 (new, not gate-blocking, low): residual FrozenDict items.**
  * (a) MappingProxyType parity lost: `reversed(fd)`, `fd.copy()` and `fd | other` raise. No current consumer uses them.
  * (b) Stored arrays are owning, so `I.gnss[1].setflags(write=True)` re-enables writes into the cached instance. This is
    the same accepted bypass class as for every Instance array.
  * (c) A caller-supplied read-only owning array is kept by reference (documented and tested). The caller can re-enable
    writes on it later.
  * (d) Non-ndarray mutable values (list, object-array cells, masked-array masks) are not frozen.
  * (e) `fd.__init__(...)` re-binds `_d` (dunder).
* **M4 risks:**
  * **R1, dense-Q cancellation persists.** At truth (bbar = b_true), z^T Q z = 1.2e-8 vs factored 6.7e-22. Use the
    factored cost for primal values and gaps; rescale before any Clarabel solve. Barfoot's sigma_1/sigma_2 >= 1e5 test
    cannot be resolved at this conditioning.
  * **R2, P2 is the only handedness guard** (re-confirmed). Dropping P2 while keeping all 8 redundant families admits
    det -1 lifted points (34 / 120 blocks at bbar = 0). Never prune P2.
  * **R3, the redundant families are linearly dependent** (472 -> 453; aniso 490 -> 471). Remove the dependencies before
    Clarabel.
  * **R4, isotropic weights are not conservative** (pass-6 C45).
  * **R5, C35's 1e-10 tolerance is sampler-fragile** (2.9e-11 here, with exactly orthogonal rotations).
  * **R7, the J finite-difference error uses part of the C24 budget** (1.1e-13 iso, 6.7e-13 at a 1e6 weight, bbar = 0
    truth).
  * **R8, results are not bit-comparable** with Rai's published tables (no MOSEK); say so wherever they are compared.
  * **R9, the surrogate gap at bbar = 0** (truth cost 3.3e-3) lies outside C26's domain.
  * **R10 (new): K != 3 is unverified.** The Layout arithmetic for other chain lengths is untested (F13 b), and the
    constraint / cost builders iterate `inst.edges` with `layout.n_kf`, `n_edges`. Rai's experiments use K = 8 / 16 / 24.
    Port the C01 / C09-C14 / C24 checks to a generic K before scaling.
  * **R11 (new): one source of truth for Dt.** The suite now pins the boundary's G to `deltaTij()`, but the cost forms
    read Dt separately and a nominal 0.5 there passes (F13 a). M4 instances with sample-count-dependent durations make
    this mismatch larger (g x dDt).
