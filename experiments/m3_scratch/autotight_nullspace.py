"""M3 scratch: AutoTight-style nullspace cross-check on the edge-0 sub-vector (67 entries of z).

Sampler: N i.i.d. feasible points: R_k ~ Exp(N(0,4 I)) (covers SO(3) incl. near-pi), v,p ~ N(0,I),
b ~ N(0, 0.09 I), liftings computed from their definitions, s = 1.  Basis: all 2278 monomials
z_a z_b (a <= b) of the sub-vector.  Nullspace of the sample matrix = every quadratic identity
valid on the lifted variety in this basis.  Compared with the span of the hand-derived families
restricted to the same entries (families touching entries outside the sub-vector are skipped).

Run:  python -m experiments.m3_scratch.autotight_nullspace
"""
import numpy as np
from .instance import build, so3_exp
from .qcqp import Problem, idx_R, idx_b, idx_e, IDX_S

rng = np.random.default_rng(4)
NAMES = ['R0'] * 9 + ['v0'] * 3 + ['p0'] * 3 + ['R1'] * 9 + ['v1'] * 3 + ['p1'] * 3 + ['b0'] * 6 + \
        ['Y'] * 9 + ['Yp'] * 9 + ['V'] * 3 + ['Vp'] * 3 + ['P'] * 3 + ['Pp'] * 3 + ['s']
CLS = {'R0': 'R', 'R1': 'R', 'v0': 'x', 'p0': 'x', 'v1': 'x', 'p1': 'x', 'b0': 'b', 's': 's'}


def main(N=6000):
    inst = build(np.zeros(6)); prob = Problem(inst)
    sub = np.r_[idx_R(0):idx_R(0) + 15, idx_R(1):idx_R(1) + 15, idx_b(0):idx_b(0) + 6, idx_e(0):idx_e(0) + 30, IDX_S]
    m = len(sub); iu = np.triu_indices(m); Z = np.zeros((N, len(iu[0])))
    for t in range(N):
        Rs = [so3_exp(rng.normal(size=3) * 2) for _ in range(3)]
        st = (Rs, [rng.normal(size=3) for _ in range(3)], [rng.normal(size=3) for _ in range(3)], [rng.normal(size=6) * 0.3 for _ in range(3)])
        zs = prob.z_of_state(*st)[sub]; Z[t] = np.outer(zs, zs)[iu]
    U, S, Vt = np.linalg.svd(Z, full_matrices=True)
    r = int((S > S[0] * 1e-10).sum()); Nsp = Vt[r:].T; P = Nsp @ Nsp.T
    print(f"sub-vector entries={m}, monomials={len(iu[0])}, samples={N}, sample rank={r}, nullspace dim={Nsp.shape[1]}")
    print("  singular values around the cut:", S[r - 2:r + 2])
    F = prob.constraints()

    def famvecs(fams):
        out = []
        for fam in fams:
            for _, A, b in F.get(fam, []):
                As = A[np.ix_(sub, sub)]
                if b != 0 or np.abs(A).sum() - np.abs(As).sum() > 1e-12:
                    continue
                C = 2 * As - np.diag(np.diag(As)); out.append(C[iu].copy())
        return np.array(out)

    stages = [("primary P1+P2+P3", ["P1_orth_cols", "P2_hand_cols", "P3_lift"]),
              ("+ Rai Table I analogues A1,B1,B2,C1", ["RA1_orth_rows", "RB1_kernel", "RB2_gram_wedge", "RC1_norm"]),
              ("+ RG cross-Gram (completes Galilean Gram)", ["RG_cross_gram"]),
              ("+ RL direct lifting", ["RL_direct"]),
              ("+ RH lifting handedness", ["RH_lift_hand"]),
              ("+ RS row-skew", ["RS_row_skew"])]
    fams = []
    for name, add in stages:
        fams += add; Vv = famvecs(fams)
        rk = np.linalg.matrix_rank(Vv, tol=1e-8); inside = np.linalg.norm(Vv - Vv @ P) / np.linalg.norm(Vv)
        print(f"  {name:45s} vectors={len(Vv):4d} rank={rk:4d} of nullspace {Nsp.shape[1]}  outside-nullspace rel={inside:.1e}")
    # what is left: restrict to monomial classes
    pairs = np.array([''.join(sorted(CLS.get(NAMES[a], 'L') + CLS.get(NAMES[b], 'L'))) for a, b in zip(*iu)])
    Vv = famvecs(fams)
    for keep in [{'LL', 'bb', 'ss'}, {'Lb'}, {'LL', 'Lb'}, {'LL', 'Lb', 'bb', 'ss', 'Ls', 'bs'}]:
        mask = np.isin(pairs, list(keep)); s_ = np.linalg.svd(Z[:, mask], compute_uv=False)
        nd = mask.sum() - int((s_ > s_[0] * 1e-10).sum())
        ok = [v for v in Vv if np.abs(v[~mask]).max() < 1e-12]
        hr = np.linalg.matrix_rank(np.array(ok), tol=1e-8) if ok else 0
        print(f"  monomial class {sorted(keep)}: nullspace {nd}, hand-derived rank {hr}")


if __name__ == "__main__":
    main()
