"""M3 scratch: is an anisotropic IMU weight a quadratic form in the BASE lifting z (n=124)?

Target on edge 0:  f_W(state) = rho_v^T W rho_v,  rho_v = (R_0 dRbar)^T r_v(z)  (the body-frame
velocity residual).  Fit f_W by least squares over ALL degree<=2 monomials of the 67 entries of z
that edge 0 touches (2278 monomials), on 6000 random feasible points.  A zero fit residual means
f_W IS a quadratic form on the feasible set; a nonzero one means it is not (in this basis)."""
import numpy as np
from .instance import build, so3_exp
from .qcqp import Problem, idx_R, idx_v, idx_p, idx_b, idx_e, IDX_S

rng = np.random.default_rng(2)


def main():
    inst = build(np.zeros(6)); prob = Problem(inst)
    sub = np.r_[idx_R(0):idx_R(0) + 15, idx_R(1):idx_R(1) + 15, idx_b(0):idx_b(0) + 6, idx_e(0):idx_e(0) + 30, IDX_S]
    m = len(sub); iu = np.triu_indices(m)
    N = 6000
    Z = np.zeros((N, len(iu[0]))); RV = np.zeros((N, 3)); RP = np.zeros((N, 3))
    for t in range(N):
        Rs = [so3_exp(rng.normal(size=3) * 2) for _ in range(3)]
        vs = [rng.normal(size=3) for _ in range(3)]; ps = [rng.normal(size=3) for _ in range(3)]
        bs = [rng.normal(size=6) * 0.3 for _ in range(3)]
        z = prob.z_of_state(Rs, vs, ps, bs)
        zs = z[sub]; Z[t] = np.outer(zs, zs)[iu]
        _, rv, rp = prob.residual_values(z, 0)
        M = Rs[0] @ prob.ed[0]["Rc"]; RV[t] = M.T @ rv; RP[t] = M.T @ rp
    print(f"monomials={Z.shape[1]} samples={N} rank(sample matrix)={np.linalg.matrix_rank(Z, tol=1e-8)}")
    Wd = {"iso tau*I": 3.0 * np.eye(3), "diag(1,2,5)": np.diag([1.0, 2.0, 5.0]),
          "near-iso diag(1,1,1.01)": np.diag([1, 1, 1.01])}
    A = np.random.default_rng(3).normal(size=(3, 3)); Wd["random SPD"] = A @ A.T + np.eye(3)
    for name, W in Wd.items():
        f = np.einsum("ti,ij,tj->t", RV, W, RV)
        c, *_ = np.linalg.lstsq(Z, f, rcond=None)
        res = np.linalg.norm(Z @ c - f) / np.linalg.norm(f)
        print(f"  W = {name:24s} relative LSQ residual = {res:.2e}")
    # cross term v-p (needed for a 6x6 block weight)
    f = np.einsum("ti,ti->t", RV, RP); c, *_ = np.linalg.lstsq(Z, f, rcond=None)
    print(f"  cross term rho_v.rho_p (W_vp off-diagonal = I)  relative residual = {np.linalg.norm(Z @ c - f) / np.linalg.norm(f):.2e}")
    f = np.einsum("ti,ti->t", RV, RV) + 0 * RP[:, 0]
    # world-frame anisotropic (fixed W applied to r_v itself, not rho_v): trivially quadratic
    _ = f


if __name__ == "__main__":
    main()
