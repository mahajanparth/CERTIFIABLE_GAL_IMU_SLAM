"""Machine-readable conventions table for the four source works (milestone M0).

Shape is fixed by the contract in ``docs/00_literature_matrix.md`` section 8.  Every entry
here is a *recorded published fact*; nothing is computed and nothing is inferred.

Three rules the tests enforce (contract section 8):

1. Every value of an ``eq`` entry is either a digit-bearing equation label (``"19"``,
   ``"22b"``, ``"37a"``) or **exactly** the string :data:`COULD_NOT_EXTRACT`.  No other
   sentinel, no ``None``, no ``""``.
2. ``eq`` labels are **per-source**.  ``rai`` Eq. (19) and ``delama`` Eq. (19) are different
   equations; the labels are namespaced by the outer source key and must never be compared
   across rows.  Equality of two labels from two sources carries no meaning whatsoever.
3. :data:`INCOMPATIBILITIES` is keyed ``"O1"``..``"O7"`` and records, for each row of
   section 3 of the matrix, the two sides and the affected coordinate blocks.

Conventions used for the structured (non-string) fields:

* ``element_order``  -- semantics of the *matrix* slots, in matrix order.
* ``ctor_order``     -- semantics of the *API constructor* arguments, in call order.
* ``tangent_order``  -- the source's own letters for the tangent coordinates.
* ``tangent_slots``  -- what those coordinates *mean*, slot by slot.  Two sources may use
  different letters for the same slot sequence (``delama`` vs ``gtsam``); comparing
  ``tangent_slots`` is meaningful, comparing ``tangent_order`` is not.
* ``None`` means "this source has no such object", not "unknown".
"""
from __future__ import annotations

#: The only permitted non-numeric value of an ``eq`` entry.  Never guess an equation number.
COULD_NOT_EXTRACT = "could not extract"

#: The four source keys, in the order the literature matrix tabulates them.
SOURCE_KEYS = ("rai", "delama", "barfoot", "gtsam")


CONVENTIONS: dict[str, dict] = {
    # ------------------------------------------------------------------ rai ---
    "rai": {
        "citation": "arXiv:2609.38048v1 [cs.RO], 29 Sep 2026 "
                    "(Rai, Xu, Chen, Rosen -- A QCQP-Representable IMU Pre-Integration "
                    "Factor for Certifiable State Estimation)",
        "source_url": "https://arxiv.org/html/2609.38048v1",
        "group": "SO(3)",
        # SO(3) per keyframe plus v, p in R^3; there is no composite group element, so the
        # per-keyframe state is 3 (rotation) + 3 (velocity) + 3 (position).
        "state_dim": 9,
        "element_order": None,   # no 5x5 (or any) group element exists in this work
        "ctor_order": None,
        "tangent_order": None,   # the only tangent object is xi_ij in R^3 (gyro-bias corr.)
        "tangent_slots": None,
        # v_i and p_i never leave R^3 (Eqs. 21b, 21c, 36) -- this is what "SO(3) only" means.
        "velocity_space": "R^3",
        "position_space": "R^3",
        "error_handedness": "none",     # chordal / Frobenius matrix residual, not a group error
        "action_handedness": "none",
        "bias_order": ("b_g", "b_a"),   # gyro first
        "bias_dim": 6,
        "bias_virtual_slices": (),
        "bias_zero_process_noise": (),
        "bias_correction_side": "none",  # absorbed algebraically via Cay(xi_ij), then lifted
        "exp_log_essential": False,      # Log removed by Eq. (14), Exp removed by Cayley Eq. (3)
        "exact": ("cayley rotation residual Eq. (19)", "denominator clearing Eq. (18)",
                  "lifting Eq. (28)"),
        "linearized": ("bias->measurement Jacobians A^{vg},A^{va},A^{pg},A^{pa} at bbar",
                       "cost weighting Eq. (23)"),

        # -- Eq. (19) / Eq. (21a), recorded structurally so the signs are decidable ------
        # r^R_ij = (R_j - R_i dRbar_ij) - 1/2 (R_j + R_i dRbar_ij) xi_ij^
        # with xi_ij = J^R_ij * delta_b_g and delta_b_g = b_g - bbar_g (the PERTURBATION).
        "rotation_residual": {
            "terms": [(+1.0, "R_j"), (-1.0, "R_i dRbar"), (-0.5, "(R_j + R_i dRbar) xi^")],
            "xi_uses": "delta_b_g",
            "xi_definition": "xi_ij = J^R_ij * delta_b_g,  delta_b_g = b_g - bbar_g",
        },

        # -- Eq. (28), the lifted decision vector, as an ordered (scope, name) sequence ---
        "lifted_vector": (
            ("keyframe", "vec(R_i)"),
            ("keyframe", "v_i"),
            ("keyframe", "p_i"),
            ("bias_block", "b_m"),
            ("imu_edge", "vec(Y_ij)"),
            ("imu_edge", "vec(Y'_ij)"),
            ("imu_edge", "W^v_ij"),
            ("imu_edge", "W^p_ij"),
            ("global", "s"),           # the homogenising scalar, LAST
        ),

        # -- Eq. (31), the primary constraint families.  All THREE are required: dropping
        #    the handedness family admits det = -1 spurious points (pre-flight F3).
        "primary_constraints": {
            "homogeniser": "s**2 == 1",
            "column_orthonormality": "R_i^T R_i == s**2 * I_3",
            "handedness": "R_i^(a) x R_i^(b) == s * R_i^(c) over cyclic (a,b,c)",
        },

        # -- Table I / Eqs. (32)-(35).  B2 has no published per-edge count. ---------------
        "redundant_constraints": {
            "A1": {"eq": "32", "cardinality": "6/node",
                   "identity": "R_k R_k^T == s**2 * I_3 (row orthonormality)"},
            "B1": {"eq": "33", "cardinality": "3/lifting",
                   "identity": "Y^(')_ij w_ij == 0,  w_ij = J^R_ij b_{g,m(i,j)}"},
            "B2": {"eq": "34", "cardinality": COULD_NOT_EXTRACT,
                   "identity": "(Y^(')_ij)^T Y^(')_ij == ||w_ij||^2 I_3 - w_ij w_ij^T"},
            "C1": {"eq": "35", "cardinality": "2/edge",
                   "identity": "||W^._ij||^2 == ||u^._ij||^2,  . in {v, p}"},
        },

        # -- Sec. VII-B ablation.  Quoted figures, not measurements taken here. -----------
        "ablation": {
            "total_solves": 840,            # 2 primitives x 7 noise levels x 12 seeds x 5 stages
            "gap_through_B1_at_least": 0.88,
            "certified_through_B1": "0/504",
            "median_gap_full_set": 2.8e-5,
            "decisive_subset": "B2",
            "instance": "synthetic K=8, M=2, n=340",
        },

        # -- Sec. IV-B.  Isotropy is forced by the device, not chosen for convenience. ----
        "isotropy_required": True,
        "isotropy_reason": "tau*||r||^2 == tau*||R_i r||^2 for R_i in SO(3): premultiplying "
                           "the velocity/position residuals by R_i is what removes the "
                           "R_i^T v_j and R_i^T p_j products, and that equality holds only "
                           "for an isotropic weight -- so the scalar precision is "
                           "structurally forced, not a tuning choice",

        # -- Sec. VII-A numerical-rank rule (NOT barfoot's; see section 5 of the matrix). --
        "rank1_criterion": {
            "kind": "eigenvalue count above a relative floor",
            "statistic": "#{lambda_i(Z_hat) > 1e-6 * lambda_max(Z_hat)}",
            "relation": "==",
            "threshold": 1,
            "relative_floor": 1e-6,
            "spectrum": "eigenvalues of Z_hat",
        },
        "tightness_degrades_with": ("stationary segments (unobservable heading)",),

        "eq": {
            "cayley": "3",
            "cayley_right_jacobian": "4",
            "qcqp": "5",
            "sdp": "6",
            "bias_ordering": COULD_NOT_EXTRACT,   # unnumbered display between (10c) and (11)
            "chordal_residual": "14",
            "xi_definition": "15",
            "cayley_determinant": "17",
            "denominator_clearing": "18",
            "rotation_residual": "19",
            "rotation_residual_restated": "21a",
            "velocity_residual": "21b",
            "position_residual": "21c",
            "lifted_rotation_residual": "22b",
            "lifting_definitions_Y": "22c",
            "isotropic_weighting": "23",
            "u_definitions": "26",
            "lifted_vector": "28",
            "lifting_constraints": "30",
            "homogeniser": "31",
            "state_set": "36",
            "rank1_criterion": COULD_NOT_EXTRACT,  # Sec. VII-A prose
        },
    },

    # --------------------------------------------------------------- delama ---
    "delama": {
        "citation": "arXiv:2411.05548v8 [cs.RO], 21 Aug 2025; RA-L, DOI 10.1109/LRA.2024.3511424 "
                    "(Delama, Fornasier, Mahony, Weiss -- Equivariant IMU Preintegration with "
                    "Biases: a Galilean Group Approach)",
        "source_url": "https://arxiv.org/html/2411.05548v8",
        # The SYMMETRY group, not the manifold.  The manifold is M = Gal(3) x R^10.
        "group": "Gal(3) |x gal(3)",
        "state_dim": 20,
        "group_dim_components": (("Gal(3)", 10), ("gal(3)", 10)),
        "manifold": "Gal(3) x R^10",
        # Eq. (3): X = [[A, a, b], [0, 1, c], [0, 0, 1]], A in SO(3), a, b in R^3, c in R.
        "element_order": ("A", "a", "b", "c"),
        "element_matrix": "[[A, a, b], [0_{1x3}, 1, c], [0_{1x3}, 0, 1]]",
        "element_slots": ("rotation", "velocity", "position", "time"),
        "ctor_order": ("dR", "dv", "dp", "dt"),   # tuple written to match the matrix
        # Eq. (4): x = (omega, v, r, alpha) in R^10 -- rot, velocity-slot, position-slot, time.
        "tangent_order": ("omega", "v", "r", "alpha"),
        "tangent_slots": ("rotation", "velocity", "position", "time"),
        "inverse_formula": "X^-1 = (A^T, -A^T a, -A^T (b - c a), -c)",
        "adjoint_time_row": (0, 0, 0, 1),     # the time coordinate is Ad-invariant
        "adjoint_time_invariant": True,
        "error_handedness": "right-invariant",      # navigation error Upsilon_k Upsilon_hat^-1
        "action_handedness": "right",               # stated immediately after Eq. (24)
        "action_handedness_citation": "the sentence immediately after Eq. (24): "
                                      "'Then, phi is a transitive and free right group "
                                      "action of G on M.'",
        "semidirect_product_law": "XY = (AB, a + Ad_A[b])",
        "semidirect_inverse_law": "X^-1 = (A^-1, -Ad_{A^-1}[a])",
        "linearised_error_dim": 20,
        "A_hat_shape": (20, 20),
        "B_hat_shape": (20, 20),
        "block_row_dims": (10, 10),
        "bias_order": ("b_omega", "b_a", "b_nu", "b_rho"),
        "bias_dim": 10,
        "bias_virtual_slices": (slice(6, 9), slice(9, 10)),
        "bias_virtual_names": ("b_nu", "b_rho"),
        "bias_virtual_kinds": ("virtual velocity bias", "virtual time bias"),
        "bias_zero_process_noise": ("b_nu", "b_rho"),
        "bias_virtual_inputs": {"nu_k": "0_{3x1}", "rho_k": "1"},
        "bias_correction_side": "left",             # Eq. (39), exp(...) applied on the left
        "exp_log_essential": True,
        "exp_log_dependencies": {
            "log_G": {"eq": "31",
                      "requires": ("log", "J_L^{-1}"),
                      "requires_eq": {"log": "9", "J_L^{-1}": "10"}},
            "error_local_coords": {"eq": "34",
                                   "requires": ("log", "J_L^{-1}"),
                                   "requires_eq": {"log": "9", "J_L^{-1}": "10"}},
        },
        "exact": ("held-input mean propagation Eq. (22)",
                  "endpoint factor Gamma_ij Eq. (18)"),
        "linearized": ("error dynamics Eqs. (35)-(37) about epsilon_k = 0",
                       "bias update Eq. (39), first order in Delta b_hat"),
        "rank1_criterion": None,            # not a certifiable formulation
        "tightness_degrades_with": (),
        "eq": {
            "semidirect_product": "1",
            "semidirect_inverse": "2",
            "element": "3",
            "tangent": "4",
            "inverse": "5",
            "adjoint": "6",
            "exp": "8",
            "log": "9",
            "left_jacobian": "10",
            "endpoint": "17",
            "gravity_left_factor": "18",
            "preintegrated_increment": "19",
            "mean_propagation": "22",
            "state_action": "24",
            "input_action": "26",
            "equivariant_error": "29",
            "local_chart": "30",
            "log_G": "31",
            "error_local_coords": "34",
            "error_dynamics": "35",
            "error_dynamics_A": "36",
            "error_dynamics_B": "37",
            "bias_correction": "39",
            "bias_process_noise": COULD_NOT_EXTRACT,  # unnumbered Define: line of Algorithm 1
            "continuous_time_ins": COULD_NOT_EXTRACT,  # cross-referenced as "( IV-A )"
        },
    },

    # -------------------------------------------------------------- barfoot ---
    "barfoot": {
        "citation": "arXiv:2308.12418v2, 31 May 2024; IJRR 44(3):366-387, 2025 "
                    "(Barfoot, Holmes, Dumbgen -- Certifiably Optimal Rotation and Pose "
                    "Estimation Based on the Cayley Map)",
        "source_url": "https://arxiv.org/html/2308.12418v2",
        "group": "SO(3) and SE(3)",
        "state_dim": 6,                       # SE(3); no velocity, no time in the state
        "element_order": ("C", "r"),          # T = [[C, r], [0, 1]], Eq. (8)
        "element_matrix": "[[C, r], [0_{1x3}, 1]]",
        "element_slots": ("rotation", "position"),
        "ctor_order": None,                   # no composite constructor
        "tangent_order": ("rho", "phi"),      # Eq. (10) -- TRANSLATION FIRST
        "tangent_slots": ("position", "rotation"),
        "error_handedness": "none",           # cost is built directly on the lifted variables
        "action_handedness": "none",
        "bias_order": None,
        "bias_dim": None,
        "bias_virtual_slices": (),
        "bias_zero_process_noise": (),
        "bias_correction_side": "none",
        "exp_log_essential": False,           # Cayley Eqs. (14)/(15) replace exp wholesale
        "exact": ("cayley lifting Eqs. (14), (15)", "homogeneous QCQP Eq. (18)"),
        "linearized": (),
        "relaxation": "Shor Eq. (20); tightness == rank(X) == 1",
        # Sec. 3 prose, restated verbatim in the captions of Figs. 2, 4, 5, 9, 12.
        "rank1_criterion": {
            "kind": "logarithmic singular-value ratio (log-SVR)",
            "statistic": "log10(sigma_1 / sigma_2)",
            "relation": ">=",
            "threshold": 5,
            "relative_floor": 1e-5,
            "spectrum": "singular values of the SDP solution X",
        },
        "tightness_degrades_with": ("measurement sparsity",),
        "tightness_degrades_evidence": "Sec. 5 discussion prose: the log SVR values are lower "
                                       "in Fig. 12 than Fig. 9 because the continuous-time "
                                       "experiments used pose measurements only at the start, "
                                       "middle and end (Holmes and Barfoot, 2023)",
        "eq": {
            "so3": "5",
            "se3_element": "8",
            "se3_tangent": "10",
            "adjoint": "11",
            "cayley": "14a",
            "cayley_inverse": "14b",
            "cayley_se3": "15",
            "nonconvex": "17",
            "homogeneous_qcqp": "18",
            "rank1_sdp": "19",
            "shor_relaxation": "20",
            "rank1_criterion": COULD_NOT_EXTRACT,   # Sec. 3 prose + figure captions
            "sparsity": COULD_NOT_EXTRACT,          # Sec. 5 discussion prose
            "licq_failure": COULD_NOT_EXTRACT,      # Sec. 6 prose
        },
    },

    # ----------------------------------------------------------------- gtsam ---
    "gtsam": {
        "citation": "GTSAM Galilean preintegration docs "
                    "borglab.github.io/gtsam/galileanimufactor/ (+ .../galileanimufactornees/), "
                    "behaviour re-verified against the installed gtsam 4.3.0 wheel",
        "source_url": "https://borglab.github.io/gtsam/galileanimufactor/",
        "package_version": "4.3.0",
        "group": "Gal(3) x R^6",                 # direct product, Eq. (1)
        "state_dim": 16,                         # 10 (Gal(3)) + 6 (ConstantBias)
        "group_dim_components": (("Gal(3)", 10), ("R^6", 6)),
        # Eq. (2): X = [[R, v, p], [0, 1, t], [0, 0, 1]] -- velocity column BEFORE position.
        "element_order": ("R", "v", "p", "t"),
        "element_matrix": "[[R, v, p], [0_{1x3}, 1, t], [0_{1x3}, 0, 1]]",
        "element_slots": ("rotation", "velocity", "position", "time"),
        # Gal3(R, r, v, t) -- position BEFORE velocity, i.e. opposite to the matrix.
        "ctor_order": ("R", "r", "v", "t"),
        "ctor_slots": ("rotation", "position", "velocity", "time"),
        # Eq. (5): x = (omega, nu, rho, alpha) in R^10 -- componentwise identical to delama (4).
        "tangent_order": ("omega", "nu", "rho", "alpha"),
        "tangent_slots": ("rotation", "velocity", "position", "time"),
        "navstate_ctor_order": ("R", "p", "v"),
        "navstate_tangent_order": ("dtheta", "dp", "dv"),
        "navstate_tangent_slots": ("rotation", "position", "velocity"),
        # Eq. (38): P_N selects Gal3 rows [0:3, 6:9, 3:6] -- a selection AND a permutation.
        "navstate_row_selection": (0, 1, 2, 6, 7, 8, 3, 4, 5),
        "navstate_block_map": (0, 6, 3),
        "navstate_block_map_symbol": COULD_NOT_EXTRACT,  # compiled wheel; behaviour confirmed
        "error_handedness": "left-invariant",    # e = xi_hat^-1 xi, Eq. (21)
        "retraction": "X (+) delta == X Exp(delta)  (right perturbation)",
        "action_handedness": "none",
        "bias_order": ("b_a", "b_g"),            # imuBias::ConstantBias -- ACCEL FIRST
        "bias_dim": 6,
        "bias_virtual_slices": (),
        "bias_zero_process_noise": (),
        "bias_correction_side": "right",         # Upsilon_hat(beta_hat) Exp(J delta_beta)
        "exp_log_essential": True,               # Exp Eq. (6); residual Eq. (42) is a logmap
        "exact": ("held-input Galilean composition Eqs. (12), (14)",
                  "endpoint relation Eq. (17)",
                  "rotating-frame relation Eq. (47) for constant rate"),
        "linearized": ("transition Eqs. (25)-(29)",
                       "bias correction Eq. (34), first order"),
        "endpoint_relation": "X_j = W_ij . phi_{dt_ij}(X_i) . U_ij(beta_i)",
        "endpoint_factors": {
            "W_ij": "NavState(I, 0.5*g*dt_ij**2, g*dt_ij)",
            "phi_dt": "phi_dt(R, p, v) = (R, p + v*dt, v)",
            "U_ij": "NavState(dR_ij, dp_ij, dv_ij)",
        },
        "endpoint_rejected_form": "H_ij X_i D_ij  (coasting map dropped)",
        "imu_sample_order": ("acc", "omega"),
        "rank1_criterion": None,                 # not certifiable, and claims none
        "tightness_degrades_with": (),
        "eq": {
            "state": "1",
            "element": "2",
            "product": "3",
            "inverse": "4",
            "tangent": "5",
            "exp": "6",
            "adjoint": "9",
            "imu_sample": "10",
            "composition": "12",
            "held_input_composition": "14",
            "gravity_left_factor": "15",
            "preintegrated_increment": "16",
            "endpoint": "17",
            "bias_lift_E": "20",
            "error": "21",
            "error_vector": "22",
            "C_matrix": "24",
            "transition": "29",
            "bias_correction": "34",
            "bias_jacobian_recursion": "35",
            "bias_correction_restated": "36",
            "navstate_tangent": "37",
            "navstate_projection": "38",
            "residual": "42",
            "rotating_frame": "47",
        },
    },
}


# The seven places where two sources (or two APIs inside one source) disagree about what
# the coordinates MEAN while agreeing about their SHAPE.  Every one produces a silently
# wrong number rather than an exception.
INCOMPATIBILITIES: dict[str, dict] = {
    "O1": {
        "where": "gtsam Gal3 matrix vs gtsam Gal3 constructor",
        "a": {"source": "gtsam", "eq": "2",
              "says": "X = [[R, v, p], [0,1,t], [0,0,1]] -- velocity in matrix column index 3, "
                      "position in column index 4"},
        "b": {"source": "gtsam", "eq": COULD_NOT_EXTRACT,
              "says": "Gal3(R, r, v, t) -- position is the 2nd argument, velocity the 3rd"},
        "blocks": ("v", "p"),
        "consequence": "velocity and position swap silently",
        "runtime_check": "Gal3(Rot3.Identity(), [1,2,3], [10,20,30], 7.0).matrix() puts "
                         "[10,20,30] in column index 3 and [1,2,3] in column index 4",
    },
    "O2": {
        "where": "gtsam Gal3 tangent vs gtsam NavState tangent",
        "a": {"source": "gtsam", "eq": "5",
              "says": "Gal3 tangent (omega, nu, rho, alpha) -- velocity block before position"},
        "b": {"source": "gtsam", "eq": "37",
              "says": "NavState tangent (dtheta, dp, dv) -- position block before velocity"},
        "blocks": ("nu", "rho"),
        "consequence": "the 9-of-10 map is a selection AND a permutation; dropping the "
                       "permutation silently swaps the velocity and position covariance blocks",
        "P_N_eq": "38",
        "P_N_rows": (0, 1, 2, 6, 7, 8, 3, 4, 5),
        "P_N_blocks": "Gal3 rows [0:3, 6:9, 3:6]",
    },
    "O3": {
        "where": "gtsam bias vs rai / delama / this project",
        "a": {"source": "gtsam", "eq": "1",
              "says": "imuBias::ConstantBias(accel, gyro); vector() == [b_a; b_omega] "
                      "-- accelerometer bias first"},
        "b": {"source": "rai", "eq": COULD_NOT_EXTRACT,
              "says": "b = [b_g; b_a] (unnumbered display after Eq. 10c); delama uses "
                      "(b_omega, b_a, b_nu, b_rho); this project fixes [b_g; b_a]"},
        "blocks": ("b_g", "b_a"),
        "consequence": "the two 3-blocks swap: accelerometer bias gets applied to the "
                       "gyroscope. One explicit permutation at the GTSAM boundary, in one place",
        "runtime_check": "imuBias.ConstantBias([1,2,3],[40,50,60]).vector() == "
                         "[1,2,3,40,50,60]",
    },
    "O4": {
        "where": "IMU sample ordering",
        "a": {"source": "gtsam", "eq": "10",
              "says": "u = [a; omega]; integrateMeasurement(measuredAcc, measuredOmega, deltaT)"},
        "b": {"source": "rai", "eq": "8a",
              "says": "Eqs. (8a)(8b) list omega_tilde then a_tilde"},
        "blocks": ("acc", "omega"),
        "consequence": "the same swap as O3 but on the measurements; the lift matrix E "
                       "(gtsam Eq. 20, R^{10x6}) is exactly this permutation",
        "runtime_check": "integrateMeasurement([1,2,3], [0,0,0], dt) leaves deltaRij == I "
                         "and sets deltaVij == [0.1,0.2,0.3] at dt = 0.1",
    },
    "O5": {
        "where": "barfoot SE(3) tangent vs everything else",
        "a": {"source": "barfoot", "eq": "10",
              "says": "xi = [rho; phi] -- translation first"},
        "b": {"source": "gtsam", "eq": "5",
              "says": "gtsam and delama put rotation first in every tangent vector"},
        "blocks": ("rho", "phi"),
        "consequence": "any constraint template copied from barfoot into a Gal(3) lifting "
                       "lands in the wrong block",
    },
    "O6": {
        "where": "naming collision on the letters r and rho",
        "a": {"source": "barfoot", "eq": "10",
              "says": "rho is the TRANSLATION coordinate; and gtsam's Gal3(R, r, ...) uses r "
                      "for POSITION"},
        "b": {"source": "delama", "eq": "4",
              "says": "r is the POSITION-SLOT tangent coordinate and v the velocity-slot, so "
                      "delama's (omega, v, r, alpha) IS gtsam's (omega, nu, rho, alpha) "
                      "componentwise"},
        "blocks": ("r", "rho"),
        "consequence": "the letters agree with gtsam by luck, not by convention. "
                       "Do not infer ordering from a letter",
    },
    "O7": {
        "where": "gravity position-block sign in the endpoint relation",
        "a": {"source": "delama", "eq": "18",
              "says": "Gamma_ij = [[I, g*dt, -0.5*g*dt**2], [0,1,-dt], [0,0,1]], applied on "
                      "the LEFT of a Gal(3) product",
              "gravity_position_coefficient": -0.5},
        "b": {"source": "gtsam", "eq": "15",
              "says": "W_ij = NavState(I, +0.5*g*dt**2, g*dt), applied on the LEFT of an "
                      "SE_2(3) product AFTER the coasting map phi_dt",
              "gravity_position_coefficient": +0.5},
        "blocks": ("p",),
        "consequence": "using +0.5*g*dt**2 inside a Gal(3) left factor is off by exactly "
                       "+g*dt**2 -- this is pre-flight defect F2",
        "reason": "not a contradiction: the sign difference is exactly the Galilean t2*v1 "
                  "coupling term of the gtsam group product, eq 3. With t2 = dt and "
                  "v1 = g*dt the left Gal(3) factor re-contributes +g*dt**2, so the stored "
                  "entry must be -0.5*g*dt**2 to land on +0.5*g*dt**2. SE_2(3) has no such "
                  "coupling because it carries no time coordinate.",
    },
}
