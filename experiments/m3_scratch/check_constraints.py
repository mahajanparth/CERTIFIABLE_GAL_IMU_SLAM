"""M3 scratch: constraint counts, vanishing at truth / random states, non-triviality, Q structure."""
import numpy as np
from .instance import build, so3_exp
from .qcqp import Problem, PRIMARY, REDUNDANT, eval_family, direct_cost, idx_R, idx_b, idx_e, IDX_S, NKF

rng = np.random.default_rng(0)


def rand_state(scale_b=0.2):
    Rs = [so3_exp(rng.normal(size=3) * 2) for _ in range(NKF)]
    vs = [rng.normal(size=3) for _ in range(NKF)]; ps = [rng.normal(size=3) * 3 for _ in range(NKF)]
    bs = [rng.normal(size=6) * scale_b for _ in range(NKF)]
    return Rs, vs, ps, bs


def main():
    inst = build()
    for aniso in (False, True):
        W = [np.eye(6) for _ in range(2)]
        prob = Problem(inst, aniso=aniso, W_vp=W)
        F = prob.constraints()
        print(f"\n=== aniso={aniso}  n={prob.n}")
        X = inst["truth"]; Rs = [x[0] for x in X]; vs = [x[1] for x in X]; ps = [x[2] for x in X]; bs = [inst["b_true"]] * 3
        zt = prob.z_of_state(Rs, vs, ps, bs)
        tot_p = tot_r = 0
        for fam in PRIMARY + REDUNDANT:
            if fam not in F: continue
            res_t = np.abs(eval_family(F[fam], zt)).max()
            res_r = max(np.abs(eval_family(F[fam], prob.z_of_state(*rand_state()))).max() for _ in range(200))
            # scale-relative: |z|^2 is O(10-100)
            pert = []
            for _ in range(20):
                zp = zt + 1e-3 * rng.normal(size=prob.n)
                pert.append(np.abs(eval_family(F[fam], zp)).max())
            # rank of family (as vech vectors)
            Mv = np.array([A[np.triu_indices(prob.n)] for _, A, _ in F[fam]])
            rk = np.linalg.matrix_rank(Mv, tol=1e-9)
            print(f"{fam:16s} count={len(F[fam]):4d} rank={rk:4d}  max|g| truth={res_t:.1e} rand(200)={res_r:.1e}  min-over-pert max|g|={min(pert):.1e}")
            if fam in PRIMARY: tot_p += len(F[fam])
            else: tot_r += len(F[fam])
        allA = np.array([A[np.triu_indices(prob.n)] for fam in F for _, A, _ in F[fam]])
        print("primary total", tot_p, " redundant total", tot_r, " joint rank", np.linalg.matrix_rank(allA, tol=1e-9))
        if not aniso:
            Q = prob.Q()
            Rcols = np.concatenate([np.arange(idx_R(k), idx_R(k) + 9) for k in range(NKF)])
            bcols = np.arange(45, 63)
            print("Q symmetric", np.abs(Q - Q.T).max(), " min eig", np.linalg.eigvalsh(Q).min() / np.abs(np.linalg.eigvalsh(Q)).max())
            print("max |Q[R,b]| =", np.abs(Q[np.ix_(Rcols, bcols)]).max(), " max |Q[Y-cols,:]| =", np.abs(Q[63:72]).max())


if __name__ == "__main__":
    main()
