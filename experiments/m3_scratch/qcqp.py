"""M3 scratch: lifted vector z, constraint families {A_k, b_k}, cost Q (throwaway, not src/).

Index map (0-based, half-open), n = 124 for 3 KF / 2 edges:
  KF k in {0,1,2}:  vec(R_k) [15k, 15k+9)  (column-major, Rai Eq.(27))
                    v_k      [15k+9, 15k+12)
                    p_k      [15k+12, 15k+15)
  bias k:           b_k      [45+6k, 51+6k)   = [b_g; b_a]
  edge e=(i,i+1):   base 63+30e:  vec(Y) [+0,+9)  vec(Y') [+9,+18)  V [+18,+21)  V' [+21,+24)
                                  P [+24,+27)  P' [+27,+30)
  s:                123
  anisotropic variant (n = 136): rho_v,e [124+6e, 127+6e), rho_p,e [127+6e, 130+6e)

Linear-form calculus: a "linear form" is a length-n vector l with value l.z; a matrix of linear
forms has shape (r, c, n).  Every constraint is a sum of products of two linear forms, hence a
quadratic form z^T A z (degree <= 2 by construction); b_k = 1 only for s^2 = 1.
"""
from __future__ import annotations
import numpy as np
from .instance import sk, gal, gal_hat, gamma_left, G_W

NKF, NE = 3, 2
N_BASE = 124


def idx_R(k): return 15 * k
def idx_v(k): return 15 * k + 9
def idx_p(k): return 15 * k + 12
def idx_b(k): return 45 + 6 * k
def idx_e(e): return 63 + 30 * e
IDX_S = 123
def idx_rho(e): return 124 + 6 * e


def index_map(aniso=False):
    m = {}
    for k in range(NKF):
        m[f"R{k}"] = (idx_R(k), idx_R(k) + 9); m[f"v{k}"] = (idx_v(k), idx_v(k) + 3); m[f"p{k}"] = (idx_p(k), idx_p(k) + 3)
    for k in range(NKF):
        m[f"b{k}"] = (idx_b(k), idx_b(k) + 6)
    for e in range(NE):
        o = idx_e(e)
        for name, a, l in [("Y", 0, 9), ("Yp", 9, 9), ("V", 18, 3), ("Vp", 21, 3), ("P", 24, 3), ("Pp", 27, 3)]:
            m[f"{name}{e}"] = (o + a, o + a + l)
    m["s"] = (IDX_S, IDX_S + 1)
    if aniso:
        for e in range(NE):
            m[f"rhov{e}"] = (idx_rho(e), idx_rho(e) + 3); m[f"rhop{e}"] = (idx_rho(e) + 3, idx_rho(e) + 6)
    return m


class Lin:
    """Builders for linear forms over z (length n)."""
    def __init__(self, n): self.n = n
    def e(self, i):
        l = np.zeros(self.n); l[i] = 1.0; return l
    def vec(self, start, length):
        return np.stack([self.e(start + a) for a in range(length)])           # (length, n)
    def mat3(self, start):                                                     # column-major 3x3
        M = np.zeros((3, 3, self.n))
        for c in range(3):
            for r in range(3):
                M[r, c, start + 3 * c + r] = 1.0
        return M
    def s(self): return self.e(IDX_S)


def mm_const_left(C, M):   # C (r x k) const, M (k x c x n) -> (r x c x n)
    return np.einsum("rk,kcn->rcn", C, M)
def mm_const_right(M, C):  # M (r x k x n), C (k x c) -> (r x c x n)
    return np.einsum("rkn,kc->rcn", M, C)
def mv_const(M, c):        # M (r x k x n), c (k) const -> (r x n)
    return np.einsum("rkn,k->rn", M, c)
def mv_lin_const(C, v):    # C (r x k) const, v (k x n) -> (r x n)
    return C @ v


def quad(l1, l2):
    """symmetric A with z^T A z = (l1.z)(l2.z)"""
    A = np.outer(l1, l2); return 0.5 * (A + A.T)


def skew_lin(w):           # w (3 x n) -> (3 x 3 x n) matrix of linear forms [w]_x
    n = w.shape[1]; M = np.zeros((3, 3, n))
    M[0, 1] = -w[2]; M[0, 2] = w[1]; M[1, 0] = w[2]; M[1, 2] = -w[0]; M[2, 0] = -w[1]; M[2, 1] = w[0]
    return M


class Problem:
    """Holds the instance constants and builds z(state), constraints and cost."""

    def __init__(self, inst, aniso=False, W_vp=None):
        self.inst = inst; self.aniso = aniso
        self.n = 136 if aniso else N_BASE
        self.L = Lin(self.n)
        self.W_vp = W_vp      # list (per edge) of 6x6 PSD weights on [rho_v; rho_p] (aniso only)
        self.ed = []
        for e, E in enumerate(inst["edges"]):
            d = {}
            d["i"], d["j"] = e, e + 1
            d["Rc"] = E.dR; d["dv"] = E.dv; d["dp"] = E.dp; d["Dt"] = E.Dt
            d["JR"] = E.J[0:3, 0:3]           # gyro columns only; J[0:3,3:6] == 0 structurally
            d["Jv"] = E.J[3:6, :]; d["Jp"] = E.J[6:9, :]
            bb = inst["bbar"]
            d["wbar"] = d["JR"] @ bb[0:3]; d["cvbar"] = d["Jv"] @ bb; d["cpbar"] = d["Jp"] @ bb
            d["tR"], d["tv"], d["tp"] = E.tR, E.tv, E.tp
            self.ed.append(d)

    # ------------------------------------------------------------------ z(state)
    def z_of_state(self, Rs, vs, ps, bs, s=1.0):
        z = np.zeros(self.n)
        for k in range(NKF):
            z[idx_R(k):idx_R(k) + 9] = Rs[k].flatten(order="F"); z[idx_v(k):idx_v(k) + 3] = vs[k]; z[idx_p(k):idx_p(k) + 3] = ps[k]
            z[idx_b(k):idx_b(k) + 6] = bs[k]
        for e, d in enumerate(self.ed):
            Ri, Rj, b = Rs[d["i"]], Rs[d["j"]], bs[d["i"]]
            w = d["JR"] @ b[0:3]; cv = d["Jv"] @ b; cp = d["Jp"] @ b
            M = Ri @ d["Rc"]; o = idx_e(e)
            z[o:o + 9] = (M @ sk(w)).flatten(order="F"); z[o + 9:o + 18] = (Rj @ sk(w)).flatten(order="F")
            z[o + 18:o + 21] = M @ cv; z[o + 21:o + 24] = Rj @ cv; z[o + 24:o + 27] = M @ cp; z[o + 27:o + 30] = Rj @ cp
        z[IDX_S] = 1.0
        if self.aniso:
            for e, d in enumerate(self.ed):
                rv, rp = self.residual_values(z, e)[1:]
                M = Rs[d["i"]] @ d["Rc"]
                z[idx_rho(e):idx_rho(e) + 3] = M.T @ rv; z[idx_rho(e) + 3:idx_rho(e) + 6] = M.T @ rp
        return s * z

    def state_of_z(self, z):
        z = z / z[IDX_S]
        Rs = [z[idx_R(k):idx_R(k) + 9].reshape(3, 3, order="F") for k in range(NKF)]
        vs = [z[idx_v(k):idx_v(k) + 3] for k in range(NKF)]; ps = [z[idx_p(k):idx_p(k) + 3] for k in range(NKF)]
        bs = [z[idx_b(k):idx_b(k) + 6] for k in range(NKF)]
        return Rs, vs, ps, bs

    # ------------------------------------------------------------------ linear-form blocks
    def blocks(self, e):
        L = self.L; d = self.ed[e]; i, j = d["i"], d["j"]; o = idx_e(e)
        B = dict(Ri=L.mat3(idx_R(i)), Rj=L.mat3(idx_R(j)), vi=L.vec(idx_v(i), 3), vj=L.vec(idx_v(j), 3),
                 pi=L.vec(idx_p(i), 3), pj=L.vec(idx_p(j), 3), b=L.vec(idx_b(i), 6),
                 Y=L.mat3(o), Yp=L.mat3(o + 9), V=L.vec(o + 18, 3), Vp=L.vec(o + 21, 3), P=L.vec(o + 24, 3), Pp=L.vec(o + 27, 3))
        B["M"] = mm_const_right(B["Ri"], d["Rc"])                 # R_i dRbar
        B["w"] = d["JR"] @ B["b"][0:3]                            # omega(b) = J_R b_g  (3 x n)
        B["cv"] = d["Jv"] @ B["b"]; B["cp"] = d["Jp"] @ B["b"]
        if self.aniso:
            B["rhov"] = L.vec(idx_rho(e), 3); B["rhop"] = L.vec(idx_rho(e) + 3, 3)
        return B

    # ------------------------------------------------------------------ residuals (affine in z)
    def residual_forms(self, e):
        """r_R (3x3xn), r_v (3xn), r_p (3xn): the cleared Galilean-Cayley matrix residual
        r = A (I - x^/2) - B (I + x^/2), A = Gamma^{-1} X_j, B = X_i Ubar, x = J (b - bbar)."""
        d = self.ed[e]; B = self.blocks(e); s = self.L.s(); I = np.eye(3)
        Gm = sk(d["wbar"]); Dt = d["Dt"]
        rR = mm_const_right(B["Rj"], I + Gm / 2) - mm_const_right(B["Ri"], d["Rc"] @ (I - Gm / 2)) - 0.5 * (B["Y"] + B["Yp"])
        sum_rot_cv = mv_const(B["Rj"], d["cvbar"]) + mv_const(B["M"], d["cvbar"])
        sum_rot_cp = mv_const(B["Rj"], d["cpbar"]) + mv_const(B["M"], d["cpbar"])
        rv = B["vj"] - B["vi"] - np.outer(G_W * Dt, s) - mv_const(B["Ri"], d["dv"]) + 0.5 * sum_rot_cv - 0.5 * (B["V"] + B["Vp"])
        rp = (B["pj"] - B["pi"] - Dt * B["vi"] - np.outer(0.5 * G_W * Dt**2, s) - mv_const(B["Ri"], d["dp"])
              + 0.5 * sum_rot_cp - 0.5 * (B["P"] + B["Pp"]))
        return rR, rv, rp

    def residual_values(self, z, e):
        rR, rv, rp = self.residual_forms(e)
        return rR @ z, rv @ z, rp @ z

    # ------------------------------------------------------------------ cost
    def cost_terms(self):
        """List of (name, list-of-(weight-matrix W, rows-of-linear-forms F)) with cost = sum (Fz)^T W (Fz)."""
        L = self.L; s = L.s(); inst = self.inst; terms = []
        for e, d in enumerate(self.ed):
            rR, rv, rp = self.residual_forms(e)
            FR = rR.reshape(9, self.n)
            terms.append((f"imu{e}_R", 0.5 * d["tR"] * np.eye(9), FR))
            if self.aniso:
                B = self.blocks(e)
                terms.append((f"imu{e}_vp", self.W_vp[e], np.vstack([B["rhov"], B["rhop"]])))
            else:
                terms.append((f"imu{e}_v", d["tv"] * np.eye(3), rv))
                terms.append((f"imu{e}_p", d["tp"] * np.eye(3), rp))
        for k, y in inst["gnss"].items():
            terms.append((f"gnss{k}", inst["w_gnss"] * np.eye(3), L.vec(idx_p(k), 3) - np.outer(y, s)))
        wR, wv, wp = inst["w_prior"]; pr = inst["prior"]
        R0 = np.stack([L.e(idx_R(0) + a) for a in range(9)])            # vec(R0) column-major
        terms.append(("prior_R", 0.5 * wR * np.eye(9), R0 - np.outer(pr["R"].flatten(order="F"), s)))
        terms.append(("prior_v", wv * np.eye(3), L.vec(idx_v(0), 3) - np.outer(pr["v"], s)))
        terms.append(("prior_p", wp * np.eye(3), L.vec(idx_p(0), 3) - np.outer(pr["p"], s)))
        for e in range(NE):
            terms.append((f"rw{e}", inst["lam_rw"][e], L.vec(idx_b(e + 1), 6) - L.vec(idx_b(e), 6)))
        return terms

    def Q(self):
        Q = np.zeros((self.n, self.n))
        for _, W, F in self.cost_terms():
            Q += F.T @ W @ F
        return 0.5 * (Q + Q.T)

    # ------------------------------------------------------------------ constraints
    def constraints(self):
        """dict family -> list of (label, A, b).  Families P* primary, R* redundant."""
        L = self.L; n = self.n; s = L.s(); F = {}
        def add(fam, label, A, b=0.0):
            F.setdefault(fam, []).append((label, A, b))
        add("P0_homog", "s^2=1", quad(s, s), 1.0)
        for k in range(NKF):
            R = L.mat3(idx_R(k))
            for a in range(3):
                for c in range(a, 3):
                    A = sum(quad(R[r, a], R[r, c]) for r in range(3)) - (a == c) * quad(s, s)
                    add("P1_orth_cols", f"R{k}^T R{k}[{a},{c}]", A)
                    A = sum(quad(R[a, r], R[c, r]) for r in range(3)) - (a == c) * quad(s, s)
                    add("RA1_orth_rows", f"R{k} R{k}^T[{a},{c}]", A)
            for (a, b_, c) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
                for comp in range(3):
                    # (R^(a) x R^(b))_comp - s R^(c)_comp
                    u, w = R[:, a], R[:, b_]; c1, c2 = (comp + 1) % 3, (comp + 2) % 3
                    A = quad(u[c1], w[c2]) - quad(u[c2], w[c1]) - quad(s, R[comp, c])
                    add("P2_hand_cols", f"R{k} col{a}x col{b_} [{comp}]", A)
                    # NOTE (M3 review): row handedness R_(a) x R_(b) = s R_(c) is NOT a separate
                    # family.  Column and row cross products are both entries of cof(R), so the
                    # row version is the same set of 9 quadratic forms {cof(R)_ij - s R_ij} as P2.
                    # Verified by check_constraints.row_handedness_is_P2().
        for e in range(NE):
            B = self.blocks(e)
            W_ = skew_lin(B["w"])                                    # [omega(b)]_x as linear forms
            for pre, Mx, Ym, Vv, Pv in [("i", B["M"], B["Y"], B["V"], B["P"]), ("j", B["Rj"], B["Yp"], B["Vp"], B["Pp"])]:
                tag = f"e{e}{pre}"
                # P3 primary lifting (Rai Eq.(30) form):  Mx^T Y = [w]_x s,  Mx^T V = c^v s,  Mx^T P = c^p s
                for a in range(3):
                    for c in range(3):
                        A = sum(quad(Mx[r, a], Ym[r, c]) for r in range(3)) - quad(W_[a, c], s)
                        add("P3_lift", f"{tag} (M^T Y)[{a},{c}]", A)
                for nm, vv, cc in [("V", Vv, B["cv"]), ("P", Pv, B["cp"])]:
                    for a in range(3):
                        A = sum(quad(Mx[r, a], vv[r]) for r in range(3)) - quad(cc[a], s)
                        add("P3_lift", f"{tag} (M^T {nm})[{a}]", A)
                # RL direct-form lifting:  s*Y = Mx [w]_x  etc. (equivalent on the variety)
                for a in range(3):
                    for c in range(3):
                        A = quad(s, Ym[a, c]) - sum(quad(Mx[a, r], W_[r, c]) for r in range(3))
                        add("RL_direct", f"{tag} sY-M[w]x [{a},{c}]", A)
                for nm, vv, cc in [("V", Vv, B["cv"]), ("P", Pv, B["cp"])]:
                    for a in range(3):
                        A = quad(s, vv[a]) - sum(quad(Mx[a, r], cc[r]) for r in range(3))
                        add("RL_direct", f"{tag} s{nm}-Mc [{a}]", A)
                # RS row-skew: Y Mx^T = [Mx w]_x  (Y Mx^T = Mx [w]_x Mx^T = [Mx w]_x for Mx in SO(3)).
                # Both sides are bilinear: (lifted x rotation) and (rotation x bias).  9 per matrix lifting.
                for a in range(3):
                    for c in range(3):
                        A = sum(quad(Ym[a, k2], Mx[c, k2]) for k2 in range(3))
                        if a != c:
                            d_ = 3 - a - c
                            eps = 1.0 if (a, c, d_) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] else -1.0
                            # [x]_x[a,c] = -eps_{acd} x_d ; x = Mx w = sum_k Mx[:,k] w_k
                            A = A + eps * sum(quad(Mx[d_, k2], B["w"][k2]) for k2 in range(3))
                        add("RS_row_skew", f"{tag} (Y M^T - [Mw]x)[{a},{c}]", A)
                # RB1 kernel: Y w = 0
                for a in range(3):
                    add("RB1_kernel", f"{tag} (Y w)[{a}]", sum(quad(Ym[a, c], B["w"][c]) for c in range(3)))
                # RB2 Gram of wedge lifting: Y^T Y = |w|^2 I - w w^T
                for a in range(3):
                    for c in range(a, 3):
                        A = sum(quad(Ym[r, a], Ym[r, c]) for r in range(3))
                        A = A - (a == c) * sum(quad(B["w"][r], B["w"][r]) for r in range(3)) + quad(B["w"][a], B["w"][c])
                        add("RB2_gram_wedge", f"{tag} (Y^TY)[{a},{c}]", A)
                # RC1 norms: |V|^2 = |c^v|^2, |P|^2 = |c^p|^2
                for nm, vv, cc in [("V", Vv, B["cv"]), ("P", Pv, B["cp"])]:
                    A = sum(quad(vv[r], vv[r]) for r in range(3)) - sum(quad(cc[r], cc[r]) for r in range(3))
                    add("RC1_norm", f"{tag} |{nm}|^2", A)
                # RG cross-Gram: Y^T V = [w]_x^T c^v,  Y^T P = [w]_x^T c^p,  V^T P = c^v.c^p
                for nm, vv, cc in [("V", Vv, B["cv"]), ("P", Pv, B["cp"])]:
                    for a in range(3):
                        A = sum(quad(Ym[r, a], vv[r]) for r in range(3)) - sum(quad(W_[r, a], cc[r]) for r in range(3))
                        add("RG_cross_gram", f"{tag} (Y^T {nm})[{a}]", A)
                A = sum(quad(Vv[r], Pv[r]) for r in range(3)) - sum(quad(B["cv"][r], B["cp"][r]) for r in range(3))
                add("RG_cross_gram", f"{tag} V.P", A)
                # RH lifting handedness: (Mx e_a) x w_k = Mx ([e_a]_x c_k)  for every lifted column w_k = Mx c_k
                cols = [(f"Ycol{c}", Ym[:, c], W_[:, c]) for c in range(3)] + [("V", Vv, B["cv"]), ("P", Pv, B["cp"])]
                for nm, wk, ck in cols:
                    for a in range(3):
                        ea = np.zeros(3); ea[a] = 1.0; body = sk(ea) @ ck          # 3 x n linear forms
                        ma = Mx[:, a]
                        for comp in range(3):
                            c1, c2 = (comp + 1) % 3, (comp + 2) % 3
                            A = quad(ma[c1], wk[c2]) - quad(ma[c2], wk[c1]) - sum(quad(Mx[comp, r], body[r]) for r in range(3))
                            add("RH_lift_hand", f"{tag} (Me{a} x {nm})[{comp}]", A)
            if self.aniso:
                rR, rv, rp = self.residual_forms(e)
                for nm, rho, rr in [("v", B["rhov"], rv), ("p", B["rhop"], rp)]:
                    for a in range(3):
                        A = sum(quad(B["M"][a, r], rho[r]) for r in range(3)) - quad(s, rr[a])
                        add("P4_resid_lift", f"e{e} M rho_{nm} = s r_{nm} [{a}]", A)
                    A = sum(quad(rho[r], rho[r]) for r in range(3)) - sum(quad(rr[r], rr[r]) for r in range(3))
                    add("RR_resid_norm", f"e{e} |rho_{nm}|^2=|r_{nm}|^2", A)
                A = sum(quad(B["rhov"][r], B["rhop"][r]) for r in range(3)) - sum(quad(rv[r], rp[r]) for r in range(3))
                add("RR_resid_norm", f"e{e} rho_v.rho_p = r_v.r_p", A)
        return F


PRIMARY = ["P0_homog", "P1_orth_cols", "P2_hand_cols", "P3_lift", "P4_resid_lift"]
REDUNDANT = ["RA1_orth_rows", "RB1_kernel", "RB2_gram_wedge", "RC1_norm", "RG_cross_gram",
             "RL_direct", "RH_lift_hand", "RS_row_skew", "RR_resid_norm"]
# Rai Table I subsets and their analogues here:  A1 -> RA1_orth_rows,  B1 -> RB1_kernel,
# B2 -> RB2_gram_wedge,  C1 -> RC1_norm.  RB2 + RG + RC1 together are the Galilean Gram identity
# [Y V P]^T [Y V P] = E^T E with E = [[w]_x, c^v, c^p] (15 per lifting side).
RAI_TABLE_I = ["RA1_orth_rows", "RB1_kernel", "RB2_gram_wedge", "RC1_norm"]
GALILEAN_EXTRA = ["RG_cross_gram", "RL_direct", "RH_lift_hand", "RS_row_skew"]


def eval_family(F, z):
    return np.array([z @ A @ z - b for _, A, b in F])


# ---------------------------------------------------------------- direct (non-lifted) reference cost
def direct_cost(prob, Rs, vs, ps, bs, aniso=False):
    """Independent path: 5x5 Gal(3) matrices, no lifting.  r = A(I - x^/2) - B(I + x^/2)."""
    inst = prob.inst; J = 0.0; parts = {}
    for e, E in enumerate(inst["edges"]):
        i, j = e, e + 1
        A = np.linalg.inv(gamma_left(E.Dt)) @ gal(Rs[j], vs[j], ps[j], 0.0)
        Bm = gal(Rs[i], vs[i], ps[i], 0.0) @ E.Ubar
        x = E.J @ (bs[i] - inst["bbar"]); xh = gal_hat(x); I5 = np.eye(5)
        r = A @ (I5 - xh / 2) - Bm @ (I5 + xh / 2)
        assert abs(r[3:, :]).max() < 1e-12
        rR, rv, rp = r[:3, :3], r[:3, 3], r[:3, 4]
        c = 0.5 * E.tR * np.sum(rR**2)
        if aniso:
            M = Rs[i] @ E.dR; rho = np.concatenate([M.T @ rv, M.T @ rp]); c += rho @ prob.W_vp[e] @ rho
        else:
            c += E.tv * rv @ rv + E.tp * rp @ rp
        parts[f"imu{e}"] = c; J += c
    for k, y in inst["gnss"].items():
        c = inst["w_gnss"] * np.sum((ps[k] - y)**2); parts[f"gnss{k}"] = c; J += c
    wR, wv, wp = inst["w_prior"]; pr = inst["prior"]
    c = 0.5 * wR * np.sum((Rs[0] - pr["R"])**2) + wv * np.sum((vs[0] - pr["v"])**2) + wp * np.sum((ps[0] - pr["p"])**2)
    parts["prior"] = c; J += c
    for e in range(NE):
        db = bs[e + 1] - bs[e]; c = db @ inst["lam_rw"][e] @ db; parts[f"rw{e}"] = c; J += c
    return J, parts
