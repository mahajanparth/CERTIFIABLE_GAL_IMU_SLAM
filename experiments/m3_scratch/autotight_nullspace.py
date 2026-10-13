"""M3 scratch: AutoTight-style nullspace discovery on the edge-0 sub-vector (67 entries of z).

Sampler: 6000 i.i.d. feasible points: R_k ~ Exp(N(0,4 I)) (covers all of SO(3) incl. near-pi),
v,p ~ N(0,I), b ~ N(0, 0.09 I), liftings computed from their definitions, s = 1.
Basis: all 2278 monomials z_a z_b (a <= b) of the sub-vector (homogeneous degree 2; s^2 plays the
constant).  Nullspace of the sample matrix = every quadratic identity valid on the variety in this
basis.  Compare with the span of the hand-derived families restricted to the same entries."""
import numpy as np
from scipy.linalg import null_space, subspace_angles
from .instance import build, so3_exp
from .qcqp import Problem, idx_R, idx_b, idx_e, IDX_S, PRIMARY, REDUNDANT

rng = np.random.default_rng(4)


def main():
    inst = build(np.zeros(6)); prob = Problem(inst)
    sub = np.r_[idx_R(0):idx_R(0) + 15, idx_R(1):idx_R(1) + 15, idx_b(0):idx_b(0) + 6, idx_e(0):idx_e(0) + 30, IDX_S]
    m = len(sub); iu = np.triu_indices(m); N = 6000
    Z = np.zeros((N, len(iu[0])))
    for t in range(N):
        Rs = [so3_exp(rng.normal(size=3) * 2) for _ in range(3)]
        st = (Rs, [rng.normal(size=3) for _ in range(3)], [rng.normal(size=3) for _ in range(3)], [rng.normal(size=6) * 0.3 for _ in range(3)])
        zs = prob.z_of_state(*st)[sub]
        O = np.outer(zs, zs); O = O + O.T - np.diag(np.diag(O))    # coefficient convention: A_ab + A_ba
        Z[t] = (np.outer(zs, zs))[iu]
    # nullspace in the "upper-triangular coefficient" convention: constraint sum_{a<=b} c_ab z_a z_b = 0
    U, S, Vt = np.linalg.svd(Z, full_matrices=True)
    tol = S[0] * 1e-10; r = int((S > tol).sum()); Nsp = Vt[r:].T
    print(f"sub-vector entries={m}, monomials={len(iu[0])}, sample rank={r}, nullspace dim={Nsp.shape[1]}")
    print("  singular values around the cut:", S[r - 2:r + 2] if r + 2 <= len(S) else S[r - 2:])
    F = prob.constraints()
    def famvecs(fams):
        out = []
        for fam in fams:
            for _, A, b in F.get(fam, []):
                As = A[np.ix_(sub, sub)]
                if np.abs(A).sum() - np.abs(As).sum() > 1e-12:  # touches entries outside the edge-0 sub-vector
                    continue
                C = 2 * As - np.diag(np.diag(As))                  # z^T A z = sum_{a<=b} C_ab z_a z_b
                v = C[iu].copy()
                if b != 0:   # s^2 = 1 is inhomogeneous: not in the homogeneous nullspace by construction
                    continue
                out.append(v)
        return np.array(out)
    P = NspProj = Nsp @ Nsp.T
    def report(name, fams):
        Vv = famvecs(fams)
        rk = np.linalg.matrix_rank(Vv, tol=1e-8)
        inside = np.linalg.norm(Vv - Vv @ P) / np.linalg.norm(Vv)
        print(f"  {name:55s} vectors={len(Vv):4d} rank={rk:4d}  (fraction of nullspace {rk}/{Nsp.shape[1]})  outside-nullspace rel={inside:.1e}")
    report("primary P1+P2+P3", ["P1_orth_cols", "P2_hand_cols", "P3_lift"])
    report("+ Rai-analogue redundant RA1,RB1,RB2,RC1", ["P1_orth_cols", "P2_hand_cols", "P3_lift", "RA1_orth_rows", "RB1_kernel", "RB2_gram_wedge", "RC1_norm"])
    report("+ RG cross-Gram", ["P1_orth_cols", "P2_hand_cols", "P3_lift", "RA1_orth_rows", "RB1_kernel", "RB2_gram_wedge", "RC1_norm", "RG_cross_gram"])
    report("+ RA2 row handedness", ["P1_orth_cols", "P2_hand_cols", "P3_lift", "RA1_orth_rows", "RB1_kernel", "RB2_gram_wedge", "RC1_norm", "RG_cross_gram", "RA2_hand_rows"])
    report("+ RL direct lifting", ["P1_orth_cols", "P2_hand_cols", "P3_lift", "RA1_orth_rows", "RB1_kernel", "RB2_gram_wedge", "RC1_norm", "RG_cross_gram", "RA2_hand_rows", "RL_direct"])
    report("+ RH lifting handedness (all families)", PRIMARY + REDUNDANT)


if __name__ == "__main__":
    main()
