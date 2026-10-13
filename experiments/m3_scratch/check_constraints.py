"""M3 scratch: constraint families -- counts, ranks, vanishing at truth / random lifted states,
non-triviality off the variety, symmetry + degree, and the row-handedness == P2 identity.

Run:  python -m experiments.m3_scratch.check_constraints
"""
import numpy as np
from .instance import build, so3_exp
from .qcqp import (Problem, PRIMARY, REDUNDANT, eval_family, idx_R, IDX_S, NKF, Lin, quad, N_BASE)

rng = np.random.default_rng(0)


def rand_state(scale_b=0.2, rng=rng):
    Rs = [so3_exp(rng.normal(size=3) * 2) for _ in range(NKF)]
    vs = [rng.normal(size=3) for _ in range(NKF)]; ps = [rng.normal(size=3) * 3 for _ in range(NKF)]
    bs = [rng.normal(size=6) * scale_b for _ in range(NKF)]
    return Rs, vs, ps, bs


def truth_state(inst):
    X = inst["truth"]
    return [x[0] for x in X], [x[1] for x in X], [x[2] for x in X], [inst["b_true"].copy() for _ in X]


def row_handedness_is_P2(n=N_BASE):
    """Row-handedness forms R_(a) x R_(b) - s R_(c) coincide with the column-handedness forms (P2)."""
    L = Lin(n); s = L.s(); worst = 0.0
    for k in range(NKF):
        R = L.mat3(idx_R(k)); col, row = {}, {}
        for (a, b_, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
            for comp in range(3):
                c1, c2 = (comp + 1) % 3, (comp + 2) % 3
                u, w = R[:, a], R[:, b_]
                col[(comp, c)] = quad(u[c1], w[c2]) - quad(u[c2], w[c1]) - quad(s, R[comp, c])
                u, w = R[a, :], R[b_, :]
                row[(c, comp)] = quad(u[c1], w[c2]) - quad(u[c2], w[c1]) - quad(s, R[c, comp])
        for key in col:
            worst = max(worst, np.abs(col[key] - row[key]).max())
    return worst


def main():
    print("row-handedness forms minus P2 forms, max |difference| =", row_handedness_is_P2())
    for bbar_label, bbar in [("bbar=0", np.zeros(6)), ("bbar=b_true", None)]:
        inst = build(np.zeros(6) if bbar is not None else build().get("b_true"))
        for aniso in (False, True):
            prob = Problem(inst, aniso=aniso, W_vp=[np.eye(6)] * 2)
            F = prob.constraints()
            print(f"\n=== {bbar_label} aniso={aniso}  n={prob.n}")
            zt = prob.z_of_state(*truth_state(inst))
            tot_p = tot_r = 0; sym = 0.0; deg = 0.0
            for fam in PRIMARY + REDUNDANT:
                if fam not in F: continue
                res_t = np.abs(eval_family(F[fam], zt)).max()
                res_r = max(np.abs(eval_family(F[fam], prob.z_of_state(*rand_state()))).max() for _ in range(200))
                pert = min(np.abs(eval_family(F[fam], zt + 1e-3 * rng.normal(size=prob.n))).max() for _ in range(20))
                Mv = np.array([A[np.triu_indices(prob.n)] for _, A, _ in F[fam]])
                rk = np.linalg.matrix_rank(Mv, tol=1e-9)
                for _, A, b in F[fam]:
                    sym = max(sym, np.abs(A - A.T).max())
                    z = rng.normal(size=prob.n); lam = 1.7          # homogeneous degree 2: g(lam z)+b = lam^2 (g(z)+b)
                    deg = max(deg, abs(lam**2 * z @ A @ z - (lam * z) @ A @ (lam * z)))
                print(f"{fam:16s} count={len(F[fam]):4d} rank={rk:4d}  max|g| truth={res_t:.1e} "
                      f"rand(200)={res_r:.1e}  min-over-20-pert max|g|={pert:.1e}")
                if fam in PRIMARY: tot_p += len(F[fam])
                else: tot_r += len(F[fam])
            allA = np.array([A[np.triu_indices(prob.n)] for fam in F for _, A, _ in F[fam]])
            prA = np.array([A[np.triu_indices(prob.n)] for fam in PRIMARY if fam in F for _, A, _ in F[fam]])
            print(f"primary total {tot_p} (rank {np.linalg.matrix_rank(prA, tol=1e-9)})  redundant total {tot_r}  "
                  f"joint rank {np.linalg.matrix_rank(allA, tol=1e-9)}  max asym {sym:.1e}  homog-deg2 defect {deg:.1e}")
            nb1 = sum(1 for fam in F for _, A, b in F[fam] if b != 0)
            print("constraints with b != 0:", nb1, "(only s^2 = 1 expected)")


if __name__ == "__main__":
    main()
