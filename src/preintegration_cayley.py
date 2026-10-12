"""Baseline 2 -- Rai-style Cayley preintegration factor.

**FROM-PAPER REIMPLEMENTATION.**  Rai et al., arXiv:2609.38048v1.  No public code exists
for this method; everything here is transcribed from the equations in the paper and is
labelled with the equation number it came from.  Where the paper is ambiguous, the
ambiguity is carried in the code as an explicit variant rather than silently resolved --
see :data:`VARIANT_RAW` / :data:`VARIANT_REPARAM`.

The point of the method is a rotation residual that is *polynomial* in the states: the
Cayley transform is a rational map whose denominator Eq. (17) is provably non-vanishing,
so Eq. (16) can be cleared algebraically into the degree-<=2 residual Eq. (19) with no
case analysis.  That is what makes the factor QCQP-representable.

Discretization-wise this baseline shares Forster/Rai Eq. (9a)-(9c) with Baseline 1 and
differs **only** in the rotation retraction (``Cay`` instead of ``Exp``).  That is why
Tier 1 (1e-6) applies between Baselines 1 and 2, while Tier 2 (5e-3) applies between
either of them and the exact-ZOH Baseline 3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preintegration_forster import ForsterPreintegration
from .trajectory import hat

#: The literal reading of the paper: the raw tangent ``(omega_k - bbar_g) dt`` is
#: substituted straight into ``Cay``.  The spec fixes this reading.
VARIANT_RAW = "raw"
#: The Eq. (3) reading: ``Cay`` is fed the Cayley vector ``2 tan(theta/2) a``, which makes
#: ``Cay`` numerically identical to ``Exp``.
VARIANT_REPARAM = "reparam"

DEFAULT_VARIANT = VARIANT_RAW


# --------------------------------------------------------------------------------------
# Eq. (3) -- the Cayley map
# --------------------------------------------------------------------------------------


def cay(phi) -> np.ndarray:
    """Rai Eq. (3): ``Cay(phi) = (I - hat(phi)/2)^-1 (I + hat(phi)/2)``.

    With ``phi = 2 tan(theta/2) a`` this equals ``Exp(theta a)`` exactly (claim M2-C07).
    Fed a raw rotation vector ``theta a`` instead, it returns a rotation of
    ``2 arctan(theta/2)`` about the same axis -- an ``O(theta^3)`` difference.
    """
    P = hat(phi) / 2.0
    return np.linalg.solve(np.eye(3) - P, np.eye(3) + P)


def cayley_vector(phi) -> np.ndarray:
    """``theta a -> 2 tan(theta/2) a``, the Eq. (3) reparameterization.

    Singular at ``theta = pi``; ``||phi||`` is 1.09 at 1 rad, 28.2 at 3 rad and 4.3e4 at
    3.1415 rad, which is the chart restriction recorded in claim M2-C07's domain.
    """
    phi = np.asarray(phi, dtype=float)
    th = float(np.linalg.norm(phi))
    if th < 1e-12:
        return phi.copy()
    return (2.0 * np.tan(th / 2.0) / th) * phi


def cay_right_jacobian(phi) -> np.ndarray:
    """Rai Eq. (4): ``J_Cay(phi) = (I - hat(phi)/2) / (1 + ||phi||^2/4)``.

    This is the **right** Jacobian:
    ``Cay(phi + d) = Cay(phi) Exp(J_Cay(phi) d) + O(||d||^2)``.
    The left convention fails by O(1) -- claim M2-C07 part (b).
    """
    phi = np.asarray(phi, dtype=float)
    return (np.eye(3) - hat(phi) / 2.0) / (1.0 + float(phi @ phi) / 4.0)


def cay_denominator_det(xi) -> float:
    """``det(I - hat(xi)/2)``; Rai Eq. (17) says this equals ``1 + ||xi||^2/4``."""
    return float(np.linalg.det(np.eye(3) - hat(xi) / 2.0))


def cay_denominator_det_closed_form(xi) -> float:
    """``1 + ||xi||^2/4``.  Strictly >= 1, hence never singular (claim M2-C08)."""
    xi = np.asarray(xi, dtype=float)
    return 1.0 + float(xi @ xi) / 4.0


# --------------------------------------------------------------------------------------
# The preintegrator -- Eq. (9a)-(9c), Cayley retraction
# --------------------------------------------------------------------------------------


class CayleyPreintegration(ForsterPreintegration):
    """Eq. (9) sums with the rotation retraction replaced by ``Cay``.

    ``variant`` selects which reading of the Cayley substitution is used; the paper does
    not say which it intends, so both stay available and the gap between them is reported
    as a bounded ambiguity rather than resolved.
    """

    def __init__(self, *args, variant: str = DEFAULT_VARIANT, **kwargs):
        if variant not in (VARIANT_RAW, VARIANT_REPARAM):
            raise ValueError(f"unknown Cayley variant {variant!r}")
        super().__init__(*args, **kwargs)
        self.variant = variant

    def retract(self, phi) -> np.ndarray:
        if self.variant == VARIANT_REPARAM:
            phi = cayley_vector(phi)
        return cay(phi)


# --------------------------------------------------------------------------------------
# Residuals -- Eq. (19), (21b), (21c), (22b)
# --------------------------------------------------------------------------------------


def rotation_from_eq16(R_i, dRbar, xi) -> np.ndarray:
    """Rai Eq. (16): ``R_j = R_i dRbar (I - hat(xi)/2)^-1 (I + hat(xi)/2)``."""
    return np.asarray(R_i) @ np.asarray(dRbar) @ cay(xi)


def rotation_residual_eq19(R_i, R_j, dRbar, xi) -> np.ndarray:
    """Rai Eq. (19): ``r^R = (R_j - R_i dRbar) - 0.5 (R_j + R_i dRbar) hat(xi)``.

    Obtained from Eq. (16) by clearing the denominator -- an **exact restatement**, not a
    truncation.  It vanishes iff Eq. (16) holds (claim M2-C09), and it is degree <= 2 in
    ``(R_i, R_j, b_g)``, which is what makes it QCQP-representable.
    """
    A = np.asarray(R_i) @ np.asarray(dRbar)
    R_j = np.asarray(R_j)
    return (R_j - A) - 0.5 * (R_j + A) @ hat(xi)


def rotation_residual_truncated(R_i, R_j, dRbar, xi) -> np.ndarray:
    """``R_j - R_i dRbar (I + hat(xi))`` -- the first-order truncation of ``Cay``.

    Kept only so claim M2-C13(b) can show it is *not* what Baseline 2 does.
    """
    A = np.asarray(R_i) @ np.asarray(dRbar)
    return np.asarray(R_j) - A @ (np.eye(3) + hat(xi))


def rotation_residual_eq22b(R_i, R_j, dRbar, J_R, b_g, bbar_g) -> np.ndarray:
    """Rai Eq. (22b), the affine lift of Eq. (19).

    ``r^R = R_j (I + Gamma/2) - R_i Rbreve - 0.5 (Y + Y')`` with
    ``Gamma = hat(J_R bbar_g)``, ``Rbreve = dRbar (I - Gamma/2)``,
    ``Y = R_i dRbar hat(J_R b_g)`` and ``Y' = R_j hat(J_R b_g)``.

    ``Y`` and ``Y'`` are auxiliary lifted variables whose purpose, as the paper states, is
    to make ``r^R`` affine.  Under ``xi = J_R (b_g - bbar_g)`` this is algebraically
    identical to Eq. (19) (claim M2-C11).
    """
    R_i, R_j, dRbar, J_R = map(np.asarray, (R_i, R_j, dRbar, J_R))
    Gamma = hat(J_R @ np.asarray(bbar_g))
    Rbreve = dRbar @ (np.eye(3) - Gamma / 2.0)
    Xi = hat(J_R @ np.asarray(b_g))
    Y = R_i @ dRbar @ Xi
    Yp = R_j @ Xi
    return R_j @ (np.eye(3) + Gamma / 2.0) - R_i @ Rbreve - 0.5 * (Y + Yp)


def velocity_residual_eq21b(R_i, v_i, v_j, dvbar, Dt, g_W,
                            A_vg=None, A_va=None, db_g=None, db_a=None) -> np.ndarray:
    """Rai Eq. (21b): ``r^v = R_i^T (v_j - v_i - g_W Dt) - dvbar - A^vg db_g - A^va db_a``.

    Unlike Eq. (19) this is **first order in db and additive**.  The asymmetry with the
    rotation residual is inherited from Forster and is intentional.
    """
    r = np.asarray(R_i).T @ (np.asarray(v_j) - np.asarray(v_i) - np.asarray(g_W) * Dt)
    r = r - np.asarray(dvbar)
    if A_vg is not None and db_g is not None:
        r = r - np.asarray(A_vg) @ np.asarray(db_g)
    if A_va is not None and db_a is not None:
        r = r - np.asarray(A_va) @ np.asarray(db_a)
    return r


def position_residual_eq21c(R_i, p_i, p_j, v_i, dpbar, Dt, g_W,
                            A_pg=None, A_pa=None, db_g=None, db_a=None) -> np.ndarray:
    """Rai Eq. (21c):
    ``r^p = R_i^T (p_j - p_i - v_i Dt - 0.5 g_W Dt^2) - dpbar - A^pg db_g - A^pa db_a``.
    """
    r = np.asarray(R_i).T @ (np.asarray(p_j) - np.asarray(p_i)
                             - np.asarray(v_i) * Dt - 0.5 * np.asarray(g_W) * Dt**2)
    r = r - np.asarray(dpbar)
    if A_pg is not None and db_g is not None:
        r = r - np.asarray(A_pg) @ np.asarray(db_g)
    if A_pa is not None and db_a is not None:
        r = r - np.asarray(A_pa) @ np.asarray(db_a)
    return r


# --------------------------------------------------------------------------------------
# Eq. (20), (23) -- isotropic precisions
# --------------------------------------------------------------------------------------


def isotropic_precision(Sigma) -> float:
    """Rai Eq. (23): ``tau^-1 = (1/3) tr Sigma``, so ``tau = 3 / tr Sigma``.

    This is structurally forced, not a convenience.  Eq. (21b)/(21c) are written in the
    body frame, i.e. premultiplied by ``R_i^T`` -- the step that removes products of
    rotation states and keeps the residual polynomial.  Moving ``R_i^T`` out of the cost
    needs ``||r||^2 = ||R_i r||^2``, which holds for the plain 2-norm and fails by O(1)
    for any non-isotropic weight (claim M2-C12).

    The paper acknowledges that this drops cross-correlations and directional weighting.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    return 3.0 / float(np.trace(Sigma))


def weighted_square(r, W=None) -> float:
    """``r' W r``, or ``||r||^2`` when ``W`` is ``None`` (the isotropic case)."""
    r = np.asarray(r, dtype=float)
    if W is None:
        return float(r @ r)
    return float(r @ np.asarray(W) @ r)


# --------------------------------------------------------------------------------------
# The factor itself -- Sec. IV-A: frozen Jacobians, exact retraction
# --------------------------------------------------------------------------------------


@dataclass
class CayleyFactor:
    """Baseline 2's edge factor.

    The paper's own words are the specification: *"Pre-integration is performed at the
    nominal bias"*, *"All Jacobians are evaluated at the nominal bias"*, *"the change is
    approximated to first order"*.  So:

    * preintegration runs **once** at ``bbar`` and is never re-linearized -- the map
      ``db_g -> xi_ij = J_R db_g`` is linear with ``J_R`` frozen;
    * but the map ``xi_ij -> rotation`` is kept **exactly** via Eq. (18), with the Cayley
      denominator cleared.  Replacing ``Cay(xi)`` by the truncation ``I + hat(xi)`` is a
      different (and worse) factor -- claim M2-C13.

    The exactness is in the retraction, not in the bias dependence.
    """

    bbar: np.ndarray
    dRbar: np.ndarray
    dvbar: np.ndarray
    dpbar: np.ndarray
    Dt: float
    J_R: np.ndarray
    A_vg: np.ndarray
    A_va: np.ndarray
    A_pg: np.ndarray
    A_pa: np.ndarray
    variant: str = DEFAULT_VARIANT

    @classmethod
    def from_edge(cls, edge, bbar, variant: str = DEFAULT_VARIANT) -> "CayleyFactor":
        from .preintegration_forster import bias_jacobians_fd, integrate_edge
        bbar = np.asarray(bbar, dtype=float)
        pim = integrate_edge(edge, bbar, cls=CayleyPreintegration, variant=variant)
        J_R, A_vg, A_va, A_pg, A_pa = bias_jacobians_fd(
            edge, bbar, cls=CayleyPreintegration, variant=variant)
        return cls(bbar=bbar, dRbar=pim.dR, dvbar=pim.dv, dpbar=pim.dp,
                   Dt=pim.dt_total, J_R=J_R, A_vg=A_vg, A_va=A_va,
                   A_pg=A_pg, A_pa=A_pa, variant=variant)

    # -- frozen linear bias map --------------------------------------------------------
    def xi(self, b_g) -> np.ndarray:
        """``xi_ij = J_R (b_g - bbar_g)``.  Linear, with ``J_R`` frozen at ``bbar``."""
        return self.J_R @ (np.asarray(b_g, dtype=float) - self.bbar[0:3])

    # -- exact Cayley retraction -------------------------------------------------------
    def predicted_rotation(self, R_i, b_g) -> np.ndarray:
        """Eq. (16) ``R_j = R_i dRbar Cay(xi)``.  **No re-linearization happens here.**"""
        return rotation_from_eq16(R_i, self.dRbar, self.xi(b_g))

    def rotation_residual(self, R_i, R_j, b_g) -> np.ndarray:
        return rotation_residual_eq19(R_i, R_j, self.dRbar, self.xi(b_g))

    def rotation_residual_truncated(self, R_i, R_j, b_g) -> np.ndarray:
        return rotation_residual_truncated(R_i, R_j, self.dRbar, self.xi(b_g))

    def rotation_residual_affine(self, R_i, R_j, b_g) -> np.ndarray:
        return rotation_residual_eq22b(R_i, R_j, self.dRbar, self.J_R,
                                       b_g, self.bbar[0:3])

    def velocity_residual(self, R_i, v_i, v_j, g_W, db=None) -> np.ndarray:
        db = np.zeros(6) if db is None else np.asarray(db, dtype=float)
        return velocity_residual_eq21b(R_i, v_i, v_j, self.dvbar, self.Dt, g_W,
                                       self.A_vg, self.A_va, db[0:3], db[3:6])

    def position_residual(self, R_i, p_i, p_j, v_i, g_W, db=None) -> np.ndarray:
        db = np.zeros(6) if db is None else np.asarray(db, dtype=float)
        return position_residual_eq21c(R_i, p_i, p_j, v_i, self.dpbar, self.Dt, g_W,
                                       self.A_pg, self.A_pa, db[0:3], db[3:6])
