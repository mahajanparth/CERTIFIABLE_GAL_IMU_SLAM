"""M0 -- one test per claim in ``docs/claims/M0.yaml``.

Every claim M0-C01..M0-C15 is bound with ``@pytest.mark.claim(...)``; that marker is the
only channel the orchestrator reads, and an unbound claim reads UNTESTED and blocks the gate.

Most M0 claims are assertions about recorded published facts, so most of the work is reading
``src.conventions``.  Four of them are deliberately *not* table lookups and would be
vacuous if they were:

* **M0-C03** -- Rai Eq. (22b) vs Eq. (19), 1e4 random draws.  If this ever fails, the lifting
  of Eq. (28) is not exact and every downstream QCQP claim is void.
* **M0-C13** -- the seven ordering incompatibilities checked against the *installed*
  gtsam 4.3.0, not against the table that records them.
* **M0-C14** -- the endpoint relation, both halves: the correct form to ~1e-15 **and** the
  ``H X D``-shaped variant visibly wrong.  Without the second half a trivially-equal ``U``
  would pass.
* **M0-C15** -- Delama Eq. (18) vs the pre-flight F2 corrected left factor, bit-exact, plus
  the reversed product shown wrong so that left-vs-right placement is actually tested.

Several otherwise-table-only claims carry a numerical falsifiability anchor as well
(C02, C04, C06, C07, C12), because a sign or a side recorded only as a string decides nothing.
"""
from __future__ import annotations

import functools
import gzip
import pathlib
import re
import urllib.error
import urllib.request

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import gtsam

from experiments.preflight.gal3_checks import gal, sk
from src.conventions import CONVENTIONS, COULD_NOT_EXTRACT, INCOMPATIBILITIES

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

_FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "rai_2609.38048v1.html.gz"
_RAI_URL = "https://arxiv.org/html/2609.38048v1"

_EQ_LABEL = re.compile(r"^\d+[a-z]?$")


@functools.lru_cache(maxsize=1)
def _rai_source_html() -> str:
    """The archived rai source text, ``<script>``/``<style>`` stripped.

    Counts are computed on the HTML, not on a MathML-stripped rendering, so LaTeX spellings
    are caught.  The cached fixture keeps this off the network; when the cache is absent and
    the fetch fails the test **FAILS** rather than skipping -- a skipped provenance check is
    indistinguishable from a passing one in the evidence file.
    """
    if _FIXTURE.exists():
        raw = gzip.decompress(_FIXTURE.read_bytes())
    else:  # pragma: no cover -- only on a fresh checkout without the fixture
        try:
            req = urllib.request.Request(
                _RAI_URL, headers={"User-Agent": "cgis-m0-literature-matrix/0.1"}
            )
            with urllib.request.urlopen(req, timeout=60) as fh:
                raw = fh.read()
        except Exception as exc:
            raise AssertionError(
                f"rai source text unavailable: no cached fixture at {_FIXTURE} and the fetch "
                f"of {_RAI_URL} failed ({exc!r}). M0-C01 must FAIL, not skip."
            ) from exc
        _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        _FIXTURE.write_bytes(gzip.compress(raw, 9))

    text = raw.decode("utf-8", "replace")
    text = re.sub(r"(?is)<script.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    return text


def _skew_batch(a: np.ndarray) -> np.ndarray:
    z = np.zeros(a.shape[0])
    return np.stack(
        [
            np.stack([z, -a[:, 2], a[:, 1]], -1),
            np.stack([a[:, 2], z, -a[:, 0]], -1),
            np.stack([-a[:, 1], a[:, 0], z], -1),
        ],
        -2,
    )


def _installed_gtsam_version() -> str:
    import importlib.metadata as md

    return md.version("gtsam")


def _pim_with_motion(n: int = 50, dt: float = 0.01):
    """A preintegrated window with nonzero acc and omega over dt_ij ~ 0.5 s."""
    params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    bias = gtsam.imuBias.ConstantBias(
        np.array([0.012, -0.021, 0.033]), np.array([0.0021, 0.0013, -0.0032])
    )
    pim = gtsam.PreintegratedImuMeasurementsG(params, bias)
    acc = np.array([0.31, -0.22, 9.93])
    omega = np.array([0.051, -0.034, 0.022])
    for _ in range(n):
        pim.integrateMeasurement(acc, omega, dt)
    return params, bias, pim


# --------------------------------------------------------------------------------------
# M0-C01
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C01")
def test_rai_group_is_so3_only_and_source_text_is_galilean_free():
    rai = CONVENTIONS["rai"]
    assert rai["group"] == "SO(3)"
    # rai has no 5x5 group element at all, hence no element ordering and no tangent ordering.
    assert rai["element_order"] is None
    assert rai["tangent_order"] is None
    # velocity and position are plain R^3 variables (Eqs. 21b, 21c, 36)
    assert rai["velocity_space"] == "R^3"
    assert rai["position_space"] == "R^3"

    html = _rai_source_html()
    assert "QCQP" in html, "fetched document is not the rai paper"

    absent = (
        "Galilean",
        "SE_2",
        "\\mathrm{SE}",   # the LaTeX spelling
        "\\\\mathrm{SE}",  # and the doubled-backslash spelling, so either reading is covered
        "SE}_{2",
        "extended pose",
        "equivariant",
        "Equivariant",
        "Delama",
        "Fornasier",
        "Weiss",
    )
    counts = {term: html.count(term) for term in absent}
    assert counts == dict.fromkeys(absent, 0), counts

    # rai DOES cite barfoot (reference [6]); Mahony appears exactly once, as a co-author of
    # reference [3] (Shonan rotation averaging), never as a Delama citation.
    assert html.count("Barfoot") >= 1
    assert html.count("Mahony") == 1


# --------------------------------------------------------------------------------------
# M0-C02
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C02")
def test_rai_rotation_residual_signs_and_bias_perturbation():
    rai = CONVENTIONS["rai"]
    assert rai["eq"]["rotation_residual"] == "19"

    rr = rai["rotation_residual"]
    assert rr["terms"] == [
        (+1, "R_j"),
        (-1, "R_i dRbar"),
        (-0.5, "(R_j + R_i dRbar) xi^"),
    ]
    assert rr["xi_uses"] == "delta_b_g"

    coeff = {name: c for c, name in rr["terms"]}
    assert coeff["R_j"] == +1               # first term, positive
    assert coeff["R_i dRbar"] == -1         # ... and the minus sign on R_i dRbar
    assert coeff["(R_j + R_i dRbar) xi^"] == -0.5   # exactly -1/2, not -1 and not +1/2
    assert rr["xi_uses"] != "b_g", "xi_ij uses the bias PERTURBATION, not the full bias"

    # Numerical anchor: the recorded coefficients must actually reproduce Eq. (19), and the
    # delta_b_g-vs-b_g distinction must change the answer (otherwise it decides nothing).
    rng = np.random.default_rng(20250219)
    Ri, Rj, dRbar = (Rotation.random(1, rng=rng).as_matrix()[0] for _ in range(3))
    JR = rng.standard_normal((3, 3))
    b_g, bbar_g = rng.standard_normal(3), rng.standard_normal(3)

    pieces = {
        "R_j": Rj,
        "R_i dRbar": Ri @ dRbar,
        "(R_j + R_i dRbar) xi^": (Rj + Ri @ dRbar) @ sk(JR @ (b_g - bbar_g)),
    }
    from_table = sum(c * pieces[name] for c, name in rr["terms"])
    literal = (Rj - Ri @ dRbar) - 0.5 * (Rj + Ri @ dRbar) @ sk(JR @ (b_g - bbar_g))
    assert np.abs(from_table - literal).max() < 1e-12

    with_full_bias = (Rj - Ri @ dRbar) - 0.5 * (Rj + Ri @ dRbar) @ sk(JR @ b_g)
    assert np.abs(literal - with_full_bias).max() > 1e-3


# --------------------------------------------------------------------------------------
# M0-C03
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C03")
def test_rai_eq22b_is_algebraically_identical_to_eq19():
    """Eq. (22b) is the SAME map as Eq. (19), not a first-order stand-in for it.

    If this fails the lifting of Eq. (28) is not exact and every downstream QCQP claim --
    M3's and M4's alike -- is void.  Hence 1e4 draws and a 1e-12 bar.
    """
    rng = np.random.default_rng(20260303)
    n = 10_000

    Ri = Rotation.random(n, rng=rng).as_matrix()
    Rj = Rotation.random(n, rng=rng).as_matrix()
    dRbar = Rotation.random(n, rng=rng).as_matrix()
    # J^R need NOT be a valid Jacobian -- the identity is pure algebra in an arbitrary 3x3.
    JR = rng.standard_normal((n, 3, 3))
    b_g = rng.standard_normal((n, 3))
    bbar_g = rng.standard_normal((n, 3))

    I3 = np.eye(3)
    Gamma = _skew_batch(np.einsum("nij,nj->ni", JR, bbar_g))          # (J^R bbar_g)^
    B = _skew_batch(np.einsum("nij,nj->ni", JR, b_g))                 # (J^R b_g)^
    xi_hat = _skew_batch(np.einsum("nij,nj->ni", JR, b_g - bbar_g))   # (J^R delta_b_g)^

    Rbreve = dRbar @ (I3 - Gamma / 2)
    Y = Ri @ dRbar @ B
    Yp = Rj @ B

    lifted = Rj @ (I3 + Gamma / 2) - Ri @ Rbreve - 0.5 * (Y + Yp)          # Eq. (22b)
    raw = (Rj - Ri @ dRbar) - 0.5 * (Rj + Ri @ dRbar) @ xi_hat             # Eq. (19)

    worst = np.abs(lifted - raw).max()
    assert worst < 1e-12, f"Eq. (22b) is not Eq. (19): max abs difference {worst:.3e}"
    # Sanity: the expressions are not both trivially zero.
    assert np.abs(raw).max() > 1e-3


# --------------------------------------------------------------------------------------
# M0-C04
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C04")
def test_rai_lifted_vector_and_homogeniser():
    rai = CONVENTIONS["rai"]
    assert rai["eq"]["lifted_vector"] == "28"
    assert rai["eq"]["homogeniser"] == "31"

    assert rai["lifted_vector"] == (
        ("keyframe", "vec(R_i)"),
        ("keyframe", "v_i"),
        ("keyframe", "p_i"),
        ("bias_block", "b_m"),
        ("imu_edge", "vec(Y_ij)"),
        ("imu_edge", "vec(Y'_ij)"),
        ("imu_edge", "W^v_ij"),
        ("imu_edge", "W^p_ij"),
        ("global", "s"),
    )
    # the homogenising scalar is LAST
    assert rai["lifted_vector"][-1] == ("global", "s")
    assert [name for _, name in rai["lifted_vector"]].count("s") == 1

    # Eq. (31) must carry all THREE families.  Omitting handedness admits det = -1 spurious
    # points (pre-flight F3).
    pc = rai["primary_constraints"]
    assert set(pc) == {"homogeniser", "column_orthonormality", "handedness"}
    assert pc["homogeniser"] == "s**2 == 1"
    assert pc["column_orthonormality"] == "R_i^T R_i == s**2 * I_3"
    assert pc["handedness"] == "R_i^(a) x R_i^(b) == s * R_i^(c) over cyclic (a,b,c)"

    # Falsifiability anchor for handedness: an improper orthogonal matrix satisfies
    # R^T R == s^2 I_3 with s = 1 yet violates the cross-product family, so dropping that
    # family really does admit det = -1 points.
    rng = np.random.default_rng(31)
    R = Rotation.random(1, rng=rng).as_matrix()[0]
    improper = R @ np.diag([1.0, 1.0, -1.0])
    s = 1.0
    assert np.abs(improper.T @ improper - s**2 * np.eye(3)).max() < 1e-12
    assert np.linalg.det(improper) < 0
    cyclic = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    violations = [
        np.linalg.norm(np.cross(improper[:, a], improper[:, b]) - s * improper[:, c])
        for a, b, c in cyclic
    ]
    assert min(violations) > 1.0
    # ... and a proper rotation satisfies every one of them.
    assert max(
        np.linalg.norm(np.cross(R[:, a], R[:, b]) - s * R[:, c]) for a, b, c in cyclic
    ) < 1e-12


# --------------------------------------------------------------------------------------
# M0-C05
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C05")
def test_rai_redundant_constraints_and_ablation():
    rc = CONVENTIONS["rai"]["redundant_constraints"]
    assert set(rc) == {"A1", "B1", "B2", "C1"}
    assert rc["A1"]["eq"] == "32"
    assert rc["B1"]["eq"] == "33"
    assert rc["B2"]["eq"] == "34"
    assert rc["C1"]["eq"] == "35"

    assert rc["A1"]["cardinality"] == "6/node"
    assert rc["B1"]["cardinality"] == "3/lifting"
    assert rc["C1"]["cardinality"] == "2/edge"
    # The paper gives no per-edge count for B2; a digit-bearing value here is fabricated.
    assert rc["B2"]["cardinality"] == "could not extract"

    ab = CONVENTIONS["rai"]["ablation"]
    assert ab["total_solves"] == 840
    assert ab["gap_through_B1_at_least"] == 0.88
    assert ab["certified_through_B1"] == "0/504"
    assert ab["median_gap_full_set"] == 2.8e-5
    # A1 + B1 alone certify nothing; B2 is what makes the difference.
    assert ab["decisive_subset"] == "B2"
    assert ab["decisive_subset"] in rc


# --------------------------------------------------------------------------------------
# M0-C06
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C06")
def test_rai_isotropic_weighting_is_structurally_forced():
    rai = CONVENTIONS["rai"]
    assert rai["eq"]["isotropic_weighting"] == "23"
    assert rai["isotropy_required"] is True
    reason = rai["isotropy_reason"]
    assert "tau*||r||^2 == tau*||R_i r||^2" in reason
    assert "SO(3)" in reason

    # The word "structurally forced" is decided numerically, not taken on trust: the
    # rotation-invariance that removes the R_i^T v_j / R_i^T p_j products holds for a
    # scalar weight and FAILS for any anisotropic SPD weight.
    rng = np.random.default_rng(23)
    n = 10_000
    R = Rotation.random(n, rng=rng).as_matrix()
    r = rng.standard_normal((n, 3))
    Rr = np.einsum("nij,nj->ni", R, r)

    M = rng.standard_normal((3, 3))
    W = M @ M.T + 3.0 * np.eye(3)                      # SPD, and not a multiple of I
    assert np.abs(W - np.trace(W) / 3.0 * np.eye(3)).max() > 1e-3

    q_plain = np.einsum("ni,ij,nj->n", r, W, r)
    q_rot = np.einsum("ni,ij,nj->n", Rr, W, Rr)
    rel = np.abs(q_plain - q_rot) / np.abs(q_plain)
    assert rel.max() > 1e-3, "an anisotropic weight must break the identity"

    tau = 2.7
    p_plain = tau * np.einsum("ni,ni->n", r, r)
    p_rot = tau * np.einsum("ni,ni->n", Rr, Rr)
    assert (np.abs(p_plain - p_rot) / np.abs(p_plain)).max() < 1e-12


# --------------------------------------------------------------------------------------
# M0-C07
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C07")
def test_delama_gal3_element_tangent_inverse_adjoint():
    d = CONVENTIONS["delama"]
    assert d["eq"]["element"] == "3"
    assert d["eq"]["tangent"] == "4"
    assert d["eq"]["inverse"] == "5"
    assert d["eq"]["adjoint"] == "6"

    assert d["element_order"] == ("A", "a", "b", "c")
    assert d["element_matrix"] == "[[A, a, b], [0_{1x3}, 1, c], [0_{1x3}, 0, 1]]"
    assert d["tangent_order"] == ("omega", "v", "r", "alpha")

    # The time coordinate is Ad-invariant: the last row of Ad is [0, 0, 0, 1].
    assert d["adjoint_time_row"] == (0, 0, 0, 1)
    assert d["adjoint_time_invariant"] is True

    # Eq. (5) cross-checked numerically on the 5x5 realisation of Eq. (3).
    rng = np.random.default_rng(56)
    n = 10_000
    A = Rotation.random(n, rng=rng).as_matrix()
    a = rng.standard_normal((n, 3))
    b = rng.standard_normal((n, 3))
    c = rng.standard_normal(n)

    X = np.tile(np.eye(5), (n, 1, 1))
    X[:, 0:3, 0:3], X[:, 0:3, 3], X[:, 0:3, 4], X[:, 3, 4] = A, a, b, c

    AT = np.transpose(A, (0, 2, 1))
    Xinv = np.tile(np.eye(5), (n, 1, 1))
    Xinv[:, 0:3, 0:3] = AT
    Xinv[:, 0:3, 3] = -np.einsum("nij,nj->ni", AT, a)
    Xinv[:, 0:3, 4] = -np.einsum("nij,nj->ni", AT, b - c[:, None] * a)
    Xinv[:, 3, 4] = -c

    assert np.abs(X @ Xinv - np.eye(5)).max() < 1e-12
    assert np.abs(Xinv @ X - np.eye(5)).max() < 1e-12

    # The letters differ between delama and gtsam; the ORDERING does not.
    g = CONVENTIONS["gtsam"]
    assert g["tangent_order"] == ("omega", "nu", "rho", "alpha")
    assert d["tangent_slots"] == g["tangent_slots"] == (
        "rotation", "velocity", "position", "time"
    )
    assert d["tangent_order"] != g["tangent_order"]


# --------------------------------------------------------------------------------------
# M0-C08
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C08")
def test_delama_symmetry_group_dim20_and_right_action():
    d = CONVENTIONS["delama"]
    assert d["group"] == "Gal(3) |x gal(3)"
    assert d["state_dim"] == 20
    assert d["eq"]["semidirect_product"] == "1"
    assert d["semidirect_product_law"] == "XY = (AB, a + Ad_A[b])"
    assert d["action_handedness"] == "right"
    assert d["eq"]["state_action"] == "24"
    assert "right group action" in d["action_handedness_citation"]
    assert "Eq. (24)" in d["action_handedness_citation"]

    # 20 is dim(Gal(3)) + dim(gal(3)) = 10 + 10, and it must stay consistent with the
    # linearised block shapes of Eqs. (35)-(37) -- not merely be the integer 20.
    assert d["group_dim_components"] == (("Gal(3)", 10), ("gal(3)", 10))
    assert sum(dim for _, dim in d["group_dim_components"]) == d["state_dim"] == 20
    assert d["linearised_error_dim"] == 20
    assert d["A_hat_shape"] == d["B_hat_shape"] == (20, 20)
    assert d["block_row_dims"] == (10, 10)
    assert sum(d["block_row_dims"]) == 20
    assert len(d["block_row_dims"]) == 2
    assert d["eq"]["error_dynamics_A"] == "36"
    assert d["eq"]["error_dynamics_B"] == "37"

    # The symmetry group is NOT the manifold.
    assert d["manifold"] == "Gal(3) x R^10"
    assert d["manifold"] != d["group"]


# --------------------------------------------------------------------------------------
# M0-C09
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C09")
def test_delama_ten_dim_bias_with_four_virtual_channels():
    d = CONVENTIONS["delama"]
    assert d["bias_order"] == ("b_omega", "b_a", "b_nu", "b_rho")
    assert d["bias_dim"] == 10
    assert d["bias_virtual_slices"] == (slice(6, 9), slice(9, 10))
    assert d["bias_zero_process_noise"] == ("b_nu", "b_rho")
    # The Q_d definition is an unnumbered "Define:" line of Algorithm 1.  Any digit-bearing
    # label here would be a fabricated citation.
    assert d["eq"]["bias_process_noise"] == "could not extract"

    # The slices name a VIRTUAL VELOCITY bias and a VIRTUAL TIME bias -- 3 + 1 = 4 channels.
    assert d["bias_virtual_names"] == ("b_nu", "b_rho")
    assert d["bias_virtual_kinds"] == ("virtual velocity bias", "virtual time bias")
    widths = [s.stop - s.start for s in d["bias_virtual_slices"]]
    assert widths == [3, 1] and sum(widths) == 4
    # ... and the slices land exactly on those names inside bias_order.
    labels = np.array(["b_omega"] * 3 + ["b_a"] * 3 + ["b_nu"] * 3 + ["b_rho"])
    assert len(labels) == d["bias_dim"]
    for sl, name in zip(d["bias_virtual_slices"], d["bias_virtual_names"]):
        assert set(labels[sl]) == {name}
    assert d["bias_virtual_inputs"] == {"nu_k": "0_{3x1}", "rho_k": "1"}

    # The two bias models cannot be silently interchanged.
    g = CONVENTIONS["gtsam"]
    assert g["bias_dim"] == 6
    assert g["bias_virtual_slices"] == ()
    assert g["bias_dim"] != d["bias_dim"]


# --------------------------------------------------------------------------------------
# M0-C10
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C10")
def test_exp_log_essential_column():
    column = {k: CONVENTIONS[k]["exp_log_essential"] for k in ("rai", "barfoot", "delama", "gtsam")}
    assert column == {"rai": False, "barfoot": False, "delama": True, "gtsam": True}

    d = CONVENTIONS["delama"]
    assert d["eq"]["log_G"] == "31"
    assert d["eq"]["error_local_coords"] == "34"

    deps = d["exp_log_dependencies"]
    for label in ("log_G", "error_local_coords"):
        entry = deps[label]
        assert entry["eq"] == d["eq"][label]
        assert set(entry["requires"]) == {"log", "J_L^{-1}"}
        assert entry["requires_eq"]["log"] == "9"
        assert entry["requires_eq"]["J_L^{-1}"] == "10"

    # Falsifiability anchor for the two False entries: rai removes Log via the chordal
    # residual AND Exp via Cayley.  Removing only one leaves a non-polynomial problem.
    rai = CONVENTIONS["rai"]
    assert rai["eq"]["chordal_residual"] == "14"
    assert rai["eq"]["cayley"] == "3"


# --------------------------------------------------------------------------------------
# M0-C11
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C11")
def test_error_handedness_and_bias_correction_side_disagree_on_both_axes():
    d, g = CONVENTIONS["delama"], CONVENTIONS["gtsam"]

    assert d["error_handedness"] == "right-invariant"
    assert d["eq"]["error_local_coords"] == "34"
    assert d["bias_correction_side"] == "left"
    assert d["eq"]["bias_correction"] == "39"

    assert g["error_handedness"] == "left-invariant"
    assert g["eq"]["error"] == "21"
    assert g["bias_correction_side"] == "right"
    assert g["eq"]["bias_correction"] == "34"

    assert d["error_handedness"] != g["error_handedness"]
    assert d["bias_correction_side"] != g["bias_correction_side"]

    # Eq-label collision guard.  The label "34" appears in BOTH sources meaning entirely
    # different equations; each is read out of its own per-source dict and the two are
    # ALLOWED to be equal strings without that implying any relationship whatsoever.
    assert d["eq"]["error_local_coords"] == g["eq"]["bias_correction"] == "34"
    assert "error_local_coords" != "bias_correction"   # different facts, same label text
    assert d["citation"] != g["citation"]              # ... in different documents

    # The ERROR handedness and the state ACTION handedness are different statements.
    assert d["action_handedness"] == "right"
    assert d["eq"]["state_action"] == "24"


# --------------------------------------------------------------------------------------
# M0-C12
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C12")
def test_barfoot_log_svr_and_rai_eigenvalue_rules_are_distinct():
    b = CONVENTIONS["barfoot"]
    r = CONVENTIONS["rai"]

    bc = b["rank1_criterion"]
    assert bc["statistic"] == "log10(sigma_1 / sigma_2)"
    assert bc["relation"] == ">="
    assert bc["threshold"] == 5                 # base-10 log of a SINGULAR-value ratio, i.e. 1e5
    assert bc["spectrum"] == "singular values of the SDP solution X"
    assert b["eq"]["rank1_criterion"] == "could not extract"

    assert "measurement sparsity" in b["tightness_degrades_with"]
    assert b["eq"]["sparsity"] == "could not extract"

    rc = r["rank1_criterion"]
    assert rc["relative_floor"] == 1e-6
    assert rc["spectrum"] == "eigenvalues of Z_hat"
    assert rc != bc, "the two numerical-rank tests must not be aliased to one another"

    # Numerical anchor, with the disagreement in the RIGHT direction.  Rai cuts at 1e-6
    # relative, Barfoot at 1e-5, so they differ on ratios between 1e5 and 1e6.
    rng = np.random.default_rng(12)
    eigs = np.array([1.0, 10 ** -5.5] + [1e-9] * 8)
    Q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    Z = Q @ np.diag(eigs) @ Q.T
    Z = (Z + Z.T) / 2

    sv = np.linalg.svd(Z, compute_uv=False)
    log_svr = np.log10(sv[0] / sv[1])
    assert log_svr == pytest.approx(5.5, abs=1e-6)
    barfoot_says_rank1 = bool(log_svr >= bc["threshold"])

    lam = np.sort(np.abs(np.linalg.eigvalsh(Z)))[::-1]
    rai_rank = int((lam > rc["relative_floor"] * lam.max()).sum())
    rai_says_rank1 = bool(rai_rank == rc["threshold"])

    assert barfoot_says_rank1 is True          # log-SVR 5.5 >= 5
    assert rai_rank == 2                       # 10**-5.5 = 3.2e-6 > 1e-6 * lambda_max
    assert rai_says_rank1 is False
    assert barfoot_says_rank1 != rai_says_rank1


# --------------------------------------------------------------------------------------
# M0-C13
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C13")
def test_seven_ordering_incompatibilities_against_installed_gtsam():
    assert set(INCOMPATIBILITIES) == {"O1", "O2", "O3", "O4", "O5", "O6", "O7"}
    for key, row in INCOMPATIBILITIES.items():
        assert row["a"]["source"] in CONVENTIONS and row["b"]["source"] in CONVENTIONS, key
        assert row["blocks"], key          # affected coordinate blocks are recorded

    assert _installed_gtsam_version() == "4.3.0"

    # -- O1: matrix column order vs constructor argument order ---------------------------
    X = gtsam.Gal3(
        gtsam.Rot3.Identity(), np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]), 7.0
    )
    M = X.matrix()
    assert np.allclose(M[0:3, 3], [10.0, 20.0, 30.0])   # velocity (3rd ctor arg) in column 3
    assert np.allclose(M[0:3, 4], [1.0, 2.0, 3.0])      # position (2nd ctor arg) in column 4
    assert M[3, 4] == 7.0
    assert CONVENTIONS["gtsam"]["element_order"] == ("R", "v", "p", "t")
    assert CONVENTIONS["gtsam"]["ctor_order"] == ("R", "r", "v", "t")
    assert INCOMPATIBILITIES["O1"]["blocks"] == ("v", "p")

    # -- O2: Gal3 tangent vs NavState tangent, and the P_N permutation -------------------
    for i in (3, 4, 5):
        e10 = np.zeros(10)
        e10[i] = 1.0
        hat = gtsam.Gal3.Hat(e10)
        assert np.linalg.norm(hat[:, 3]) > 0.5          # fills the VELOCITY column
        assert np.linalg.norm(hat[:, 4]) < 1e-12

        e9 = np.zeros(9)
        e9[i] = 1.0
        moved = gtsam.NavState().retract(e9)
        assert np.linalg.norm(moved.position()) > 0.5   # ... but moves POSITION
        assert np.linalg.norm(moved.velocity()) < 1e-12

    rows = INCOMPATIBILITIES["O2"]["P_N_rows"]
    assert rows == (0, 1, 2, 6, 7, 8, 3, 4, 5)
    assert CONVENTIONS["gtsam"]["navstate_row_selection"] == rows
    assert INCOMPATIBILITIES["O2"]["P_N_eq"] == "38"

    # kGal3BlockForNavState is a C++ symbol in a compiled wheel, so the NAME cannot be
    # confirmed here -- only the behaviour.  That behaviour is tested directly: the recorded
    # row selection is the permutation that makes a Gal3 tangent step agree with a NavState
    # tangent step to second order, and the un-permuted selection does not.
    assert CONVENTIONS["gtsam"]["navstate_block_map_symbol"] == COULD_NOT_EXTRACT
    assert CONVENTIONS["gtsam"]["navstate_block_map"] == (0, 6, 3)
    assert tuple(rows[0:3]) == (0, 1, 2)
    assert tuple(rows[3:6]) == (6, 7, 8)
    assert tuple(rows[6:9]) == (3, 4, 5)

    rng = np.random.default_rng(38)
    eps = 1e-6
    Xg = gtsam.Gal3(
        gtsam.Rot3.Expmap(np.array([0.2, -0.1, 0.4])),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.4, -0.5, 0.6]),
        0.0,
    )
    NS = gtsam.NavState(Xg.rotation(), Xg.position(), Xg.velocity())
    worst_pn, worst_identity = 0.0, 0.0
    for _ in range(20):
        delta = np.zeros(10)
        delta[:9] = rng.standard_normal(9) * eps
        Xr = Xg.retract(delta)
        target = gtsam.NavState(Xr.rotation(), Xr.position(), Xr.velocity())
        worst_pn = max(
            worst_pn, np.linalg.norm(target.localCoordinates(NS.retract(delta[list(rows)])))
        )
        worst_identity = max(
            worst_identity,
            np.linalg.norm(target.localCoordinates(NS.retract(delta[:9]))),
        )
    assert worst_pn < 1e-3 * eps          # second order: the permutation is right
    assert worst_identity > 1e-1 * eps    # first order: dropping it is wrong

    # -- O3: imuBias::ConstantBias is (accel, gyro) --------------------------------------
    cb = gtsam.imuBias.ConstantBias(np.array([1.0, 2.0, 3.0]), np.array([40.0, 50.0, 60.0]))
    assert np.allclose(cb.vector(), [1.0, 2.0, 3.0, 40.0, 50.0, 60.0])
    assert np.allclose(cb.accelerometer(), [1.0, 2.0, 3.0])
    assert np.allclose(cb.gyroscope(), [40.0, 50.0, 60.0])
    assert CONVENTIONS["gtsam"]["bias_order"] == ("b_a", "b_g")
    assert CONVENTIONS["rai"]["bias_order"] == ("b_g", "b_a")      # this project's ordering
    assert CONVENTIONS["gtsam"]["bias_order"] != CONVENTIONS["rai"]["bias_order"]

    # -- O4: integrateMeasurement takes (acc, omega) -------------------------------------
    params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    zero = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    pim = gtsam.PreintegratedImuMeasurementsG(params, zero)
    pim.integrateMeasurement(np.array([1.0, 2.0, 3.0]), np.zeros(3), 0.1)
    assert np.abs(pim.deltaRij().matrix() - np.eye(3)).max() < 1e-12   # 1st arg is NOT omega
    assert np.allclose(pim.deltaVij(), [0.1, 0.2, 0.3])                # 1st arg IS acc
    swapped = gtsam.PreintegratedImuMeasurementsG(params, zero)
    swapped.integrateMeasurement(np.zeros(3), np.array([1.0, 2.0, 3.0]), 0.1)
    assert np.abs(swapped.deltaRij().matrix() - np.eye(3)).max() > 1e-3  # 2nd arg IS omega
    assert np.linalg.norm(swapped.deltaVij()) < 1e-12
    assert CONVENTIONS["gtsam"]["imu_sample_order"] == ("acc", "omega")

    # -- O5: barfoot SE(3) tangent is [rho; phi], translation first ----------------------
    assert CONVENTIONS["barfoot"]["tangent_order"] == ("rho", "phi")
    assert CONVENTIONS["barfoot"]["tangent_slots"] == ("position", "rotation")
    assert CONVENTIONS["barfoot"]["eq"]["se3_tangent"] == "10"
    assert CONVENTIONS["barfoot"]["tangent_slots"][0] == "position"
    assert CONVENTIONS["gtsam"]["tangent_slots"][0] == "rotation"
    assert CONVENTIONS["delama"]["tangent_slots"][0] == "rotation"

    # -- O6: the letters r / rho mean position in gtsam and barfoot, the position SLOT in
    #        delama -- and delama's (omega, v, r, alpha) IS gtsam's (omega, nu, rho, alpha).
    o6 = INCOMPATIBILITIES["O6"]
    assert o6["blocks"] == ("r", "rho")
    g_ctor = CONVENTIONS["gtsam"]
    assert g_ctor["ctor_order"][g_ctor["ctor_slots"].index("position")] == "r"
    bt = CONVENTIONS["barfoot"]
    assert bt["tangent_order"][bt["tangent_slots"].index("position")] == "rho"
    dt_ = CONVENTIONS["delama"]
    assert dt_["tangent_order"][dt_["tangent_slots"].index("position")] == "r"
    assert dt_["tangent_slots"] == g_ctor["tangent_slots"]

    # -- O7: gravity position-block sign ---------------------------------------------------
    o7 = INCOMPATIBILITIES["O7"]
    sign_gal3 = o7["a"]["gravity_position_coefficient"]
    sign_navstate = o7["b"]["gravity_position_coefficient"]
    assert sign_gal3 == -0.5 and sign_navstate == +0.5
    assert sign_gal3 != sign_navstate
    assert o7["a"]["eq"] == "18" and o7["b"]["eq"] == "15"
    # It is a consequence of the group law, not a contradiction: the reason must name the
    # Galilean t2*v1 coupling term of GTSAM eq 3.
    reason = o7["reason"]
    assert "t2*v1" in reason
    assert "eq 3" in reason
    assert "not a contradiction" in reason


# --------------------------------------------------------------------------------------
# M0-C14
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C14")
def test_gtsam_endpoint_relation_is_W_phi_X_U_and_not_H_X_D():
    g = CONVENTIONS["gtsam"]
    assert g["eq"]["endpoint"] == "17"
    assert g["endpoint_relation"] == "X_j = W_ij . phi_{dt_ij}(X_i) . U_ij(beta_i)"
    assert g["eq"]["gravity_left_factor"] == "15"
    assert g["eq"]["preintegrated_increment"] == "16"
    assert g["endpoint_factors"]["W_ij"] == "NavState(I, 0.5*g*dt_ij**2, g*dt_ij)"
    assert g["endpoint_factors"]["phi_dt"] == "phi_dt(R, p, v) = (R, p + v*dt, v)"
    assert g["endpoint_factors"]["U_ij"] == "NavState(dR_ij, dp_ij, dv_ij)"

    params, bias, pim = _pim_with_motion(n=50, dt=0.01)
    dt_ij = pim.deltaTij()
    assert dt_ij == pytest.approx(0.5, abs=1e-9)

    g_vec = params.n_gravity
    assert np.allclose(g_vec, [0.0, 0.0, -9.81])

    # A nontrivial X_i WITH nonzero velocity -- otherwise the coasting map phi_dt is the
    # identity and the wrong form would pass by accident.
    X_i = gtsam.NavState(
        gtsam.Rot3.Expmap(np.array([0.11, -0.23, 0.31])),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, -0.4, 0.3]),
    )
    assert np.linalg.norm(X_i.velocity()) > 0.1

    X_j = pim.predict(X_i, bias)

    W = gtsam.NavState(gtsam.Rot3.Identity(), 0.5 * g_vec * dt_ij**2, g_vec * dt_ij)
    phi = gtsam.NavState(
        X_i.attitude(), X_i.position() + X_i.velocity() * dt_ij, X_i.velocity()
    )
    U = gtsam.NavState(pim.deltaRij(), pim.deltaPij(), pim.deltaVij())

    correct = W.compose(phi).compose(U)
    err_correct = np.linalg.norm(X_j.localCoordinates(correct))
    assert err_correct <= 1e-12, f"W . phi_dt(X_i) . U does not reproduce predict(): {err_correct:.3e}"

    # ... and the coasting-free, H X D-shaped variant must be visibly WRONG.  Without this
    # half a trivially-equal U would pass the test.
    wrong = W.compose(X_i).compose(U)
    err_wrong = np.linalg.norm(X_j.localCoordinates(wrong))
    assert err_wrong > 1e-2, f"W . X_i . U is not distinguishable from the truth: {err_wrong:.3e}"
    assert g["endpoint_rejected_form"] == "H_ij X_i D_ij  (coasting map dropped)"


# --------------------------------------------------------------------------------------
# M0-C15
# --------------------------------------------------------------------------------------

@pytest.mark.claim("M0-C15")
def test_delama_eq18_gamma_equals_T_minus_dt_times_Hprime_exactly():
    """Delama Eq. (18) IS the pre-flight F2 corrected left factor -- bit-identically.

    Exactness (not a tolerance) is expected because both expressions are built from the same
    literals.  The reversed product is checked too, because a test of the first half alone
    would pass for any commuting pair and would say nothing about side placement.
    """
    assert INCOMPATIBILITIES["O7"]["a"]["eq"] == "18"
    assert CONVENTIONS["delama"]["eq"]["gravity_left_factor"] == "18"

    rng = np.random.default_rng(1815)
    worst_reversed = np.inf
    for _ in range(200):
        dt = float(rng.uniform(0.05, 2.0)) * rng.choice([-1.0, 1.0])
        g = rng.standard_normal(3)
        g *= rng.uniform(1.0, 20.0) / np.linalg.norm(g)
        assert dt != 0.0 and np.linalg.norm(g) > 0.0

        # Delama Eq. (18), written out directly.
        Gamma = np.eye(5)
        Gamma[0:3, 3] = g * dt
        Gamma[0:3, 4] = -0.5 * g * dt**2
        Gamma[3, 4] = -dt

        # Pre-flight F2 forms.
        Hp = gal(np.eye(3), g * dt, -0.5 * g * dt**2, 0.0)
        T = gal(np.eye(3), np.zeros(3), np.zeros(3), -dt)

        assert np.array_equal(Gamma, T @ Hp)
        assert np.linalg.norm(Gamma - T @ Hp) == 0.0

        # Side placement is load-bearing: on the right, T corrupts the position block.
        reversed_err = np.linalg.norm(Gamma - Hp @ T)
        worst_reversed = min(worst_reversed, reversed_err)
        assert reversed_err > 1e-6

    assert worst_reversed > 1e-6

    # Spot check with the exact values the claim quotes: dt = 0.5, g = 9.81 -> 2.45.
    dt, g = 0.5, np.array([0.0, 0.0, -9.81])
    Gamma = np.eye(5)
    Gamma[0:3, 3] = g * dt
    Gamma[0:3, 4] = -0.5 * g * dt**2
    Gamma[3, 4] = -dt
    Hp = gal(np.eye(3), g * dt, -0.5 * g * dt**2, 0.0)
    T = gal(np.eye(3), np.zeros(3), np.zeros(3), -dt)
    assert np.linalg.norm(Gamma - T @ Hp) == 0.0
    assert np.linalg.norm(Gamma - Hp @ T) == pytest.approx(9.81 * dt**2, rel=1e-12)


# --------------------------------------------------------------------------------------
# Structural invariants of the table itself (contract section 8, rules 1-3).
# Not bound to a numbered claim: it guards the shape every other test reads.
# --------------------------------------------------------------------------------------

def test_every_eq_entry_is_a_label_or_the_exact_sentinel():
    def check(where: str, mapping: dict) -> None:
        for label, value in mapping.items():
            assert isinstance(value, str), f"{where}.{label} is not a string"
            assert value == COULD_NOT_EXTRACT or _EQ_LABEL.match(value), (
                f"{where}.{label} == {value!r} is neither a digit-bearing equation label "
                f"nor exactly {COULD_NOT_EXTRACT!r}"
            )

    assert set(CONVENTIONS) == {"rai", "delama", "barfoot", "gtsam"}
    for key, row in CONVENTIONS.items():
        check(f"CONVENTIONS[{key!r}].eq", row["eq"])
        for required in (
            "citation", "group", "state_dim", "element_order", "ctor_order", "tangent_order",
            "error_handedness", "action_handedness", "bias_order", "bias_dim",
            "bias_virtual_slices", "bias_zero_process_noise", "bias_correction_side",
            "exp_log_essential", "exact", "linearized", "eq",
        ):
            assert required in row, f"CONVENTIONS[{key!r}] is missing {required!r}"
        assert row["error_handedness"] in {"left-invariant", "right-invariant", "none"}
        assert row["action_handedness"] in {"left", "right", "none"}
        assert row["bias_correction_side"] in {"left", "right", "none"}
        assert isinstance(row["exp_log_essential"], bool)
        assert isinstance(row["bias_virtual_slices"], tuple)

    for key, row in INCOMPATIBILITIES.items():
        for side in ("a", "b"):
            check(f"INCOMPATIBILITIES[{key!r}][{side!r}]", {"eq": row[side]["eq"]})
            assert row[side]["says"]
        assert isinstance(row["blocks"], tuple) and row["blocks"]
        assert row["where"] and row["consequence"]
