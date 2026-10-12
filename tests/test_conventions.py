"""M0 -- one test per claim in ``docs/claims/M0.yaml``.

Every claim M0-C01..M0-C15 is bound with ``@pytest.mark.claim(...)``; that marker is the
only channel the orchestrator reads, and an unbound claim reads UNTESTED and blocks the gate.

REMEDIATION NOTE (see ``docs/verification/m0-literature-matrix.md``, findings F-1..F-7).
--------------------------------------------------------------------------------------
The first version of this file restated the contents of :mod:`src.conventions` as literals
in the test body and then compared the table against those literals.  An independent
mutation run showed that 40 of 41 deliberately corrupted table fields still passed.  The
rule now observed throughout:

    **No expectation is written in this file.**  Every assertion reads the value out of
    ``CONVENTIONS`` / ``INCOMPATIBILITIES`` and decides it against one of four sources that
    a mutation of the table cannot move:

    1. ``docs/claims/M0.yaml`` -- the claim statements themselves, parsed.  A statement that
       says ``CONVENTIONS["rai"]["eq"]["rotation_residual"] == "19"`` *is* the expectation;
       :func:`assert_claim_statement` extracts every such predicate and checks it.
    2. The **archived primary sources** under ``tests/fixtures`` -- the four works' own
       LaTeX, keyed by the authors' own equation numbers.  Every recorded ``eq`` label is
       looked up in the source it cites and the equation found there must carry the content
       the label is recorded for (:data:`EQ_SIGNATURES`).
    3. The **installed gtsam 4.3.0**, exercised at runtime.
    4. **Execution of the recorded string itself.**  Where the table records a formula
       (``inverse_formula``, ``primary_constraints``, ``endpoint_factors``, the ``says``
       entries of the incompatibility register, ...) the test parses that string and runs
       it.  A corrupted formula then produces a wrong number rather than passing a string
       comparison against a copy of itself.

    ``docs/00_literature_matrix.md`` -- the deriver's cited matrix, a separate artefact from
    ``src/conventions.py`` -- is the corroboration source for free prose that has no numeric
    content (citations, ``where`` / ``consequence`` text).

Assertions that could not fail were deleted outright rather than reworded; they are listed
in F-7 of the verification report.
"""
from __future__ import annotations

import ast
import functools
import gzip
import html as _html
import math
import operator
import pathlib
import re
import urllib.error
import urllib.request

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

import gtsam

from experiments.preflight.gal3_checks import gal, sk
from src.conventions import CONVENTIONS, COULD_NOT_EXTRACT, INCOMPATIBILITIES

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_FIXTURES = pathlib.Path(__file__).parent / "fixtures"

_EQ_LABEL = re.compile(r"^\d+[a-z]?$")


# ======================================================================================
# Expectation source 1 -- the claim file
# ======================================================================================

@functools.lru_cache(maxsize=1)
def claims() -> dict[str, dict]:
    doc = yaml.safe_load((_ROOT / "docs" / "claims" / "M0.yaml").read_text())
    return {c["id"]: c for c in doc["claims"]}


def claim(cid: str) -> dict:
    return claims()[cid]


_CONV_PREFIX = re.compile(r'CONVENTIONS\s*\[\s*"(\w+)"\s*\]')
_LVALUE_CHAIN = re.compile(r'(?:CONVENTIONS\s*\[\s*"(\w+)"\s*\]\s*)?((?:\[\s*"[^"\]]+"\s*\]\s*)+)$')
_LVALUE_NAME = re.compile(r"(\b[A-Za-z_]\w*)\s*$")
_RVALUE = re.compile(
    r"\s*(?:"
    r'"(?P<str>[^"]*)"'
    r"|(?P<tuple>\((?:[^()]|\([^()]*\))*\))"
    r"|(?P<const>True|False|None)"
    r"|(?P<num>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r")"
)

#: Bare names used in a claim statement that are not top-level fields of the source row.
#: Only the *path* is resolved here -- the expected value still comes from the statement.
_STATEMENT_PATHS = {
    ("M0-C05", "A1"): ("rai", "redundant_constraints", "A1", "cardinality"),
    ("M0-C05", "B1"): ("rai", "redundant_constraints", "B1", "cardinality"),
    ("M0-C05", "C1"): ("rai", "redundant_constraints", "C1", "cardinality"),
    ("M0-C05", "B2"): ("rai", "redundant_constraints", "B2", "cardinality"),
    ("M0-C05", "total_solves"): ("rai", "ablation", "total_solves"),
    ("M0-C05", "gap_through_B1_at_least"): ("rai", "ablation", "gap_through_B1_at_least"),
    ("M0-C05", "certified_through_B1"): ("rai", "ablation", "certified_through_B1"),
    ("M0-C05", "median_gap_full_set"): ("rai", "ablation", "median_gap_full_set"),
    ("M0-C05", "decisive_subset"): ("rai", "ablation", "decisive_subset"),
}


def statement_expectations(cid: str) -> list[tuple[tuple[str, ...], object]]:
    """Every ``<table path> == <literal>`` predicate written into the claim's statement."""
    stmt = claim(cid)["statement"]
    found: list[tuple[tuple[str, ...], object]] = []
    for m in re.finditer(r"(==|\bis\b)", stmt):
        head, tail = stmt[: m.start()], stmt[m.end():]
        rv = _RVALUE.match(tail)
        if rv is None:
            continue
        if rv.group("str") is not None:
            expected: object = rv.group("str")
        elif rv.group("tuple") is not None:
            expected = eval(rv.group("tuple"), {"__builtins__": {}}, {"slice": slice})
        elif rv.group("const") is not None:
            expected = {"True": True, "False": False, "None": None}[rv.group("const")]
        else:
            expected = ast.literal_eval(rv.group("num"))

        prefixes = list(_CONV_PREFIX.finditer(head))
        src_ctx = prefixes[-1].group(1) if prefixes else None

        chain = _LVALUE_CHAIN.search(head)
        if chain and chain.group(2).strip():
            keys = re.findall(r'\[\s*"([^"\]]+)"\s*\]', chain.group(2))
            src = chain.group(1) or src_ctx
            if src is not None:
                found.append(((src, *keys), expected))
            continue

        name = _LVALUE_NAME.search(head.rstrip())
        if not name:
            continue
        token = name.group(1)
        if (cid, token) in _STATEMENT_PATHS:
            found.append((_STATEMENT_PATHS[(cid, token)], expected))
        elif src_ctx is not None and token in CONVENTIONS[src_ctx]:
            found.append(((src_ctx, token), expected))
    return found


def table_value(path):
    node = CONVENTIONS if path[0] in CONVENTIONS else INCOMPATIBILITIES
    for key in path:
        node = node[key]
    return node


def assert_claim_statement(cid: str) -> int:
    """Check every table predicate the claim's own statement spells out.

    The expected values are read out of ``docs/claims/M0.yaml``; nothing is written here.
    """
    exp = statement_expectations(cid)
    assert exp, f"{cid}: no table predicate could be parsed out of the claim statement"
    for path, expected in exp:
        got = table_value(path)
        assert got == expected, (
            f"{cid}: CONVENTIONS{''.join(f'[{k!r}]' for k in path)} == {got!r}, but the "
            f"claim statement in docs/claims/M0.yaml says {expected!r}"
        )
    return len(exp)


def claim_number(cid: str, field: str, pattern: str) -> float:
    """Pull a numeric regime bound out of the claim's own text."""
    m = re.search(pattern, claim(cid)[field])
    assert m, f"{cid}: {pattern!r} is not pinned in the claim's {field!r} field"
    return float(m.group(1))


# ======================================================================================
# Expectation source 2 -- the archived primary sources
# ======================================================================================

SOURCE_FIXTURES = {
    "rai": ("rai_2609.38048v1.html.gz", "https://arxiv.org/html/2609.38048v1"),
    "delama": ("delama_2411.05548v8.html.gz", "https://arxiv.org/html/2411.05548v8"),
    "barfoot": ("barfoot_2308.12418v2.html.gz", "https://arxiv.org/html/2308.12418v2"),
    "gtsam": ("gtsam_galileanimufactor.html.gz",
              "https://borglab.github.io/gtsam/galileanimufactor/"),
}


@functools.lru_cache(maxsize=8)
def source_html(key: str) -> str:
    """The archived source text for ``key``, ``<script>``/``<style>`` stripped.

    The cached fixture keeps this off the network; when the cache is absent and the fetch
    fails the test **FAILS** rather than skipping -- a skipped provenance check is
    indistinguishable from a passing one in the evidence file.
    """
    name, url = SOURCE_FIXTURES[key]
    path = _FIXTURES / name
    if path.exists():
        raw = gzip.decompress(path.read_bytes())
    else:  # pragma: no cover -- only on a checkout without the fixtures
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "cgis-m0-literature-matrix/0.1"}
            )
            with urllib.request.urlopen(req, timeout=60) as fh:
                raw = fh.read()
        except Exception as exc:
            raise AssertionError(
                f"{key} source text unavailable: no cached fixture at {path} and the fetch "
                f"of {url} failed ({exc!r}). The claim must FAIL, not skip."
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(raw, 9))
    text = raw.decode("utf-8", "replace")
    text = re.sub(r"(?is)<script.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?</style>", " ", text)
    return text


_LATEXML_BLOCK = re.compile(r'<table\b[^>]*\bid="(S[^"]*?\.E([0-9]*)[^"]*)"[^>]*>')
_LATEXML_TAG = re.compile(r"ltx_tag_equation(?:group)?[^>]*>\(([^<)]*)\)<")
_LATEXML_ALT = re.compile(r'alttext="([^"]*)"')
_MYST_MATH = re.compile(r'"type":"math"(.{0,40000}?)"enumerator":"(\d+[a-z]?)"', re.S)
_MYST_VALUE = re.compile(r'"value":"((?:[^"\\]|\\.)*)"', re.S)


@functools.lru_cache(maxsize=8)
def source_equations(key: str) -> dict[str, str]:
    """``{equation label -> the authors' own LaTeX}`` for one archived source.

    arXiv renders (rai, delama, barfoot) are LaTeXML: the number is in ``ltx_tag_equation``
    and the LaTeX in the ``alttext`` of the same block.  The GTSAM doc page is ``mystmd``
    and embeds its own AST, with ``enumerator`` next to the raw LaTeX ``value``.  Both are
    the sources' own numbering, not a rendered glyph stream.

    Sub-numbered equations whose tag LaTeXML drops (barfoot's ``(14a)``) stay reachable
    through the block's base number, which is in the block id.
    """
    raw = source_html(key)
    index: dict[str, str] = {}
    if key == "gtsam":
        for m in _MYST_MATH.finditer(raw):
            vm = _MYST_VALUE.search(m.group(1))
            if vm:
                index.setdefault(
                    m.group(2), vm.group(1).encode().decode("unicode_escape", "replace")
                )
        return index
    for m in _LATEXML_BLOCK.finditer(raw):
        base, start = m.group(2), m.end()
        depth, i = 1, start
        while depth and i < len(raw):
            nxt, close = raw.find("<table", i), raw.find("</table>", i)
            if close == -1:
                break
            if nxt != -1 and nxt < close:
                depth, i = depth + 1, nxt + 6
            else:
                depth, i = depth - 1, close + 8
        body = raw[start:i]
        latex = " ".join(_html.unescape(a) for a in _LATEXML_ALT.findall(body))
        for tag in _LATEXML_TAG.findall(body):
            index.setdefault(tag, latex)
        if base:
            index.setdefault(base, latex)
    return index


def equation_latex(key: str, label: str) -> str:
    """The LaTeX the source itself prints at ``label``.  Falls back to the block base."""
    index = source_equations(key)
    if label in index:
        return index[label]
    base = re.sub(r"[a-z]+$", "", label)
    assert base and base in index, (
        f"{key} has no equation labelled ({label}) -- the archived source "
        f"{SOURCE_FIXTURES[key][1]} numbers {len(index)} equations and this is not one"
    )
    return index[base]


def assert_equation_says(key: str, label: str, require, forbid=()) -> None:
    latex = equation_latex(key, label)
    for pattern in require:
        assert re.search(pattern, latex), (
            f"{key} Eq. ({label}) does not contain {pattern!r}; the source prints\n"
            f"  {' '.join(latex.split())[:400]}"
        )
    for pattern in forbid:
        assert not re.search(pattern, latex), (
            f"{key} Eq. ({label}) unexpectedly contains {pattern!r}"
        )


# ======================================================================================
# Expectation source 3 -- the deriver's literature matrix (corroboration for free prose)
# ======================================================================================

@functools.lru_cache(maxsize=1)
def matrix_doc() -> str:
    return (_ROOT / "docs" / "00_literature_matrix.md").read_text()


@functools.lru_cache(maxsize=1)
def _matrix_tokens() -> set[str]:
    return set(re.findall(r"[a-z0-9]+", matrix_doc().lower()))


def corroboration(text: str) -> float:
    """Fraction of a prose field's distinctive words that the cited matrix also uses."""
    words = {w for w in re.findall(r"[a-z0-9]{4,}", text.lower())}
    if not words:
        return 1.0
    doc = _matrix_tokens()
    return sum(1 for w in words if w in doc) / len(words)


def assert_corroborated(where: str, text: str, floor: float = 0.55) -> None:
    got = corroboration(text)
    assert got >= floor, (
        f"{where}: only {got:.0%} of this text's words occur anywhere in "
        f"docs/00_literature_matrix.md -- it is not the fact the matrix recorded.\n  {text!r}"
    )


_EQ_MENTION = re.compile(r"\bEqs?\.?\s*((?:\(?\d+[a-z]?\)?(?:\s*[-,;]\s*)?)+)", re.I)


def assert_equation_mentions_resolve(where: str, text: str, keys) -> None:
    """Any ``Eq. (n)`` written into a prose field must be a real equation of a cited source."""
    for run in _EQ_MENTION.findall(text):
        for label in re.findall(r"\d+[a-z]?", run):
            ok = False
            for key in keys:
                index = source_equations(key)
                if label in index or re.sub(r"[a-z]+$", "", label) in index:
                    ok = True
                    break
            assert ok, (
                f"{where} cites Eq. ({label}), which none of {tuple(keys)} actually numbers"
            )


# ======================================================================================
# Executing the strings the table records
# ======================================================================================

def _hat(a):
    return sk(np.asarray(a, dtype=float))


def _inv(m):
    return np.linalg.inv(m)


def _norm(a):
    return float(np.linalg.norm(a))


_TOKEN = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*(?:[eE][-+]?\d+)?|\*\*|[()+\-*/,]|\S")


def _is_scalar(tok: str, scalars) -> bool:
    return tok[0].isdigit() or tok in scalars


def mathify(text: str, scalars=()) -> str:
    """Turn a recorded formula such as ``-A^T (b - c a)`` into Python.

    Juxtaposition becomes ``@`` between matrix/vector operands and ``*`` when either side is
    a scalar; ``X^T`` becomes a transpose, ``X^-1`` an inverse, a trailing ``^`` the skew
    operator and ``||x||`` a norm.  Running the table's own string is the point: a corrupted
    sign or a swapped slot then produces a wrong *number* instead of passing a string
    comparison against a copy of itself.
    """
    s = text
    s = re.sub(r"\|\|\s*([A-Za-z_]\w*)\s*\|\|", r"NORM_\1", s)
    s = re.sub(r"([A-Za-z_]\w*)\^T", r"TR_\1", s)
    s = re.sub(r"([A-Za-z_]\w*)\^-1", r"INV_\1", s)
    s = re.sub(r"([A-Za-z_]\w*)\^(?![\w*(])", r"HAT_\1", s)
    out: list[str] = []
    for tok in _TOKEN.findall(s):
        if out:
            prev = out[-1]
            left = prev[-1].isalnum() or prev[-1] == "_" or prev == ")"
            right = tok[0].isalnum() or tok[0] == "_" or tok == "("
            if left and right:
                out.append("*" if (_is_scalar(prev, scalars) or _is_scalar(tok, scalars))
                           else "@")
        out.append(tok)
    s = " ".join(out)
    s = re.sub(r"TR_(\w+)", r"(\1).T", s)
    s = re.sub(r"INV_(\w+)", r"_inv(\1)", s)
    s = re.sub(r"HAT_(\w+)", r"_hat(\1)", s)
    s = re.sub(r"NORM_(\w+)", r"_norm(\1)", s)
    return s


def run_formula(text: str, env: dict, scalars=()):
    """Evaluate a formula recorded in the conventions table."""
    expr = mathify(text, scalars)
    ns = dict(env)
    ns.update(_inv=_inv, _hat=_hat, _norm=_norm, np=np)
    try:
        return eval(expr, {"__builtins__": {}}, ns)   # noqa: S307 -- repo-controlled string
    except Exception as exc:                          # pragma: no cover -- mutation path
        raise AssertionError(
            f"the recorded formula {text!r} does not evaluate ({expr!r}): {exc!r}"
        ) from exc


def gal_from_rows(text: str, env: dict) -> np.ndarray:
    """Build a 5x5 Gal(3) matrix from a recorded ``[[I, a, b],[0,1,c],[0,0,1]]`` string."""
    rows = [r for r in re.findall(r"\[([^][]*)\]", text) if r.strip()]
    assert len(rows) == 3, f"{text!r} is not a 3-row 5x5 block matrix"
    ns = dict(env)
    ns.update(np=np, I=np.eye(3))
    cells = [[c.strip() for c in row.split(",")] for row in rows]
    assert [len(c) for c in cells] == [3, 3, 3], f"{text!r} is not 3x3 blocks"
    M = np.zeros((5, 5))
    top = [eval(c, {"__builtins__": {}}, ns) for c in cells[0]]          # noqa: S307
    M[0:3, 0:3] = np.asarray(top[0], dtype=float).reshape(3, 3)
    M[0:3, 3] = np.asarray(top[1], dtype=float).reshape(3)
    M[0:3, 4] = np.asarray(top[2], dtype=float).reshape(3)
    for r, row in zip((3, 4), cells[1:]):
        for c, cell in zip((3, 4), row[1:]):
            M[r, c] = float(eval(cell, {"__builtins__": {}}, ns))        # noqa: S307
    return M


# ======================================================================================
# Shared helpers for the numerical anchors
# ======================================================================================

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


def _pim_with_motion(n: int, dt: float):
    """A preintegrated window with nonzero acc and omega over ``n*dt`` seconds."""
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


def _tangent_blocks(slots) -> dict[str, slice]:
    """Map recorded slot names onto their blocks of a 10-vector, in the recorded order."""
    widths = {"rotation": 3, "velocity": 3, "position": 3, "time": 1}
    out, at = {}, 0
    for slot in slots:
        out[slot] = slice(at, at + widths[slot])
        at += widths[slot]
    assert at == 10, f"recorded tangent slots {slots} do not fill R^10"
    return out


def _rand_gal(rng):
    return gal(Rotation.random(1, rng=rng).as_matrix()[0], rng.standard_normal(3),
               rng.standard_normal(3), float(rng.standard_normal()))


# ======================================================================================
# M0-C01
# ======================================================================================

@pytest.mark.claim("M0-C01")
def test_rai_group_is_so3_only_and_source_text_is_galilean_free():
    assert_claim_statement("M0-C01")          # group, element_order, tangent_order
    rai = CONVENTIONS["rai"]

    html = source_html("rai")
    assert "QCQP" in html, "the archived document is not the rai paper"

    # v_i and p_i never leave R^3.  The recorded spaces are turned into the LaTeX the paper
    # would have to print and that spelling is looked for in the archived source.
    for field in ("velocity_space", "position_space"):
        dim = re.fullmatch(r"R\^(\d+)", rai[field])
        assert dim, f"rai[{field!r}] == {rai[field]!r} is not of the form R^n"
        assert rf"\mathbf{{v}}_{{i}},\mathbf{{p}}_{{i}}\in\mathbb{{R}}^{{{dim.group(1)}}}" in html, (
            f"the rai source never declares v_i, p_i in {rai[field]!r}"
        )

    # The absent-term scan.  The terms come from the claim statement, not from this file:
    # the first version hard-coded a list containing both "\mathrm{SE}" and its
    # doubled-backslash spelling, and the second count could never fail independently of the
    # first because one string contained the other (F-7.7).
    stmt = claim("M0-C01")["statement"]
    absent_src = stmt.split("zero occurrences of each of", 1)[1].split("while", 1)[0]
    absent = tuple(dict.fromkeys(re.findall(r'"([^"]+)"', absent_src)))
    assert len(absent) >= 10, absent
    for i, term in enumerate(absent):
        for j, other in enumerate(absent):
            assert i == j or term not in other, (
                f"{term!r} is contained in {other!r}; its count cannot fail independently"
            )
    counts = {term: html.count(term) for term in absent}
    assert counts == dict.fromkeys(absent, 0), counts

    # rai DOES cite barfoot; Mahony appears once, as a co-author of reference [3].
    present = re.findall(r'"([^"]+)" occurs at least once', stmt)
    assert present == ["Barfoot"], present
    assert html.count("Barfoot") >= 1
    assert html.count("Mahony") == int(
        claim_number("M0-C01", "test_hint", r'"Mahony"\s*==\s*(\d+)')
    )


# ======================================================================================
# M0-C02
# ======================================================================================

@pytest.mark.claim("M0-C02")
def test_rai_rotation_residual_signs_and_bias_perturbation():
    assert_claim_statement("M0-C02")          # eq["rotation_residual"] == "19"
    rai = CONVENTIONS["rai"]
    rr = rai["rotation_residual"]

    # The recorded structure is EXECUTED against the formula the claim statement spells out.
    # The first version summed the table's coefficients over pieces it had itself written,
    # so the comparison was forced by the literal assertion above it (F-7.2).
    rng = np.random.default_rng(20250219)
    Ri, Rj, dRbar = (Rotation.random(1, rng=rng).as_matrix()[0] for _ in range(3))
    JR = rng.standard_normal((3, 3))
    b_g, bbar_g = rng.standard_normal(3), rng.standard_normal(3)

    # xi_ij comes out of the recorded definition, including which bias it is built from.
    lhs, rhs = (p.strip() for p in rr["xi_definition"].split("=", 1))
    assert lhs.startswith("xi_ij"), rr["xi_definition"]
    xi_expr, delta_def = (p.strip() for p in rhs.split(",", 1))
    delta_name, delta_expr = (p.strip() for p in delta_def.split("=", 1))
    assert delta_name == rr["xi_uses"], (
        f"xi_definition defines {delta_name!r} but xi_uses records {rr['xi_uses']!r}"
    )
    delta = run_formula(delta_expr, {"b_g": b_g, "bbar_g": bbar_g})
    xi = run_formula(xi_expr.replace("J^R_ij", "JR"), {"JR": JR, delta_name: delta})

    env = {"R_j": Rj, "R_i": Ri, "dRbar": dRbar, "xi": np.asarray(xi, dtype=float)}
    from_table = sum(c * np.asarray(run_formula(name, env)) for c, name in rr["terms"])

    stmt_formula = re.search(r"r\^R_ij\s*=\s*(.+?)\s*with xi_ij", claim("M0-C02")["statement"])
    assert stmt_formula, "the claim statement no longer spells the residual out"
    written = stmt_formula.group(1).replace("dRbar_ij", "dRbar").replace("xi_ij^", "xi^")
    expected = np.asarray(run_formula(written, env, scalars=("1", "2")))
    assert np.abs(from_table - expected).max() < 1e-12, (
        "the recorded terms do not reproduce the residual the claim statement writes out"
    )

    # ... and the perturbation-vs-full-bias distinction must change the answer, otherwise
    # recording "delta_b_g" decides nothing.
    full = run_formula(xi_expr.replace("J^R_ij", "JR"), {"JR": JR, delta_name: b_g})
    with_full_bias = np.asarray(
        run_formula(written, {**env, "xi": np.asarray(full, dtype=float)}, scalars=("1", "2"))
    )
    assert np.abs(expected - with_full_bias).max() > 1e-3


# ======================================================================================
# M0-C03
# ======================================================================================

@pytest.mark.claim("M0-C03")
def test_rai_eq22b_is_algebraically_identical_to_eq19():
    """Eq. (22b) is the SAME map as Eq. (19), not a first-order stand-in for it.

    If this fails the lifting of Eq. (28) is not exact and every downstream QCQP claim --
    M3's and M4's alike -- is void.  Hence 1e4 draws and a 1e-12 bar.

    The bar is RELATIVE (verification finding F-3): at ``J^R`` and ``b`` scaled by 1e4 the
    max *absolute* difference is 4.8e-7, so an absolute bar cannot support the claim's
    "arbitrary 3x3 J^R and arbitrary b_g" domain.  Normalising by the size of the residual
    keeps the bar meaningful at every scale, and the three regimes below prove it.
    """
    rai = CONVENTIONS["rai"]
    bar = 1e-12
    rng = np.random.default_rng(20260303)
    n = 10_000
    I3 = np.eye(3)

    for scale in (1.0, 1e-8, 1e4):
        Ri = Rotation.random(n, rng=rng).as_matrix()
        Rj = Rotation.random(n, rng=rng).as_matrix()
        dRbar = Rotation.random(n, rng=rng).as_matrix()
        # J^R need NOT be a valid Jacobian -- the identity is pure algebra in an arbitrary 3x3.
        JR = rng.standard_normal((n, 3, 3)) * scale
        b_g = rng.standard_normal((n, 3)) * scale
        bbar_g = rng.standard_normal((n, 3)) * scale

        Gamma = _skew_batch(np.einsum("nij,nj->ni", JR, bbar_g))          # (J^R bbar_g)^
        B = _skew_batch(np.einsum("nij,nj->ni", JR, b_g))                 # (J^R b_g)^
        xi_hat = _skew_batch(np.einsum("nij,nj->ni", JR, b_g - bbar_g))   # (J^R delta_b_g)^

        Rbreve = dRbar @ (I3 - Gamma / 2)
        Y = Ri @ dRbar @ B
        Yp = Rj @ B

        lifted = Rj @ (I3 + Gamma / 2) - Ri @ Rbreve - 0.5 * (Y + Yp)     # Eq. (22b)
        raw = (Rj - Ri @ dRbar) - 0.5 * (Rj + Ri @ dRbar) @ xi_hat        # Eq. (19)

        magnitude = np.maximum(np.abs(raw).max(axis=(1, 2)), 1.0)
        worst = float((np.abs(lifted - raw).max(axis=(1, 2)) / magnitude).max())
        assert worst < bar, (
            f"Eq. (22b) is not Eq. (19) at scale {scale:g}: relative difference {worst:.3e}"
        )
        # Sanity: the expressions are not both trivially zero.
        assert np.abs(raw).max() > 1e-3 * min(1.0, scale)

    # The two forms are the ones the table cites, and both equations exist in the paper.
    assert_equation_says("rai", rai["eq"]["lifted_rotation_residual"],
                         require=[r"\\breve\{\\mathbf\{R\}\}_\{ij\}"])
    assert_equation_says("rai", rai["eq"]["rotation_residual"],
                         require=[r"\\mathbf\{r\}\^\{R\}_\{ij\}\\triangleq"])


# ======================================================================================
# M0-C04
# ======================================================================================

@pytest.mark.claim("M0-C04")
def test_rai_lifted_vector_and_homogeniser():
    assert_claim_statement("M0-C04")          # eq["lifted_vector"], eq["homogeniser"]
    rai = CONVENTIONS["rai"]

    # The block sequence the claim statement lists, in order, with "s" LAST.
    stmt = claim("M0-C04")["statement"]
    scope = {"keyframe": "keyframe", "bias block": "bias_block", "IMU edge": "imu_edge"}
    expected = [
        (scope[kind], name.strip())
        for group, kind in re.findall(r"\[([^]]+)\] per (keyframe|bias block|IMU edge)", stmt)
        for name in group.split(",")
    ]
    expected.append(("global", "s"))
    assert [tuple(x) for x in rai["lifted_vector"]] == expected, (
        f"lifted_vector records {rai['lifted_vector']} but the claim statement lists {expected}"
    )

    # Eq. (31) must carry all THREE families, and every recorded constraint STRING is turned
    # into a predicate and run.  The first version asserted a closed-form fact about improper
    # rotations that read nothing out of the table (F-7.4).
    pc = rai["primary_constraints"]
    assert set(pc) == {"homogeniser", "column_orthonormality", "handedness"}
    rng = np.random.default_rng(31)
    R = Rotation.random(1, rng=rng).as_matrix()[0]
    improper = R @ np.diag([1.0, 1.0, -1.0])
    assert np.linalg.det(improper) < 0

    def residual(text: str, Rm: np.ndarray, s: float) -> float:
        cyclic = "cyclic" in text
        lhs, rhs = (p.strip() for p in text.split(" over ")[0].split("==", 1))
        env = {"s": float(s), "I_3": np.eye(3)}
        scal = ("s", "1", "2", "3")
        if cyclic:
            worst = 0.0
            for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                sub = {"A_": Rm[:, a], "B_": Rm[:, b], "C_": Rm[:, c]}
                le = lhs.replace("R_i^(a)", "A_").replace("R_i^(b)", "B_")
                ri = rhs.replace("R_i^(c)", "C_")
                assert " x " in le, f"the recorded handedness family is not a cross product: {text!r}"
                left = np.cross(sub["A_"], sub["B_"])
                right = np.asarray(run_formula(ri, {**env, **sub}, scal))
                worst = max(worst, float(np.abs(left - right).max()))
            return worst
        env["R_i"] = Rm
        left = np.asarray(run_formula(lhs, env, scal))
        right = np.asarray(run_formula(rhs, env, scal))
        return float(np.abs(left - right).max())

    assert residual(pc["homogeniser"], R, 1.0) < 1e-12
    assert residual(pc["homogeniser"], R, 2.0) > 1.0
    # Column orthonormality alone admits the improper matrix -- which is exactly why the
    # handedness family is required (pre-flight F3), and the recorded handedness string is
    # what has to reject it.
    assert residual(pc["column_orthonormality"], improper, 1.0) < 1e-12
    assert residual(pc["column_orthonormality"], 2.0 * R, 1.0) > 1.0
    assert residual(pc["handedness"], R, 1.0) < 1e-12
    assert residual(pc["handedness"], improper, 1.0) > 1.0


# ======================================================================================
# M0-C05
# ======================================================================================

@pytest.mark.claim("M0-C05")
def test_rai_redundant_constraints_and_ablation():
    assert_claim_statement("M0-C05")      # the four cardinalities and the five ablation values
    rai = CONVENTIONS["rai"]
    rc = rai["redundant_constraints"]

    # id -> equation exactly as the claim statement maps them.
    mapping = dict(re.findall(r'(\w+)->"(\d+[a-z]?)"', claim("M0-C05")["statement"]))
    assert set(mapping) == set(rc), (set(mapping), set(rc))
    for key, label in mapping.items():
        assert rc[key]["eq"] == label, (
            f"redundant_constraints[{key!r}]['eq'] == {rc[key]['eq']!r}, claim says {label!r}"
        )

    # The ablation count is not taken on trust: the claim's domain states the design and the
    # recorded total must be its product.
    domain = claim("M0-C05")["domain"]
    factors = [
        int(x) for x in
        re.findall(r"(\d+)\s*(?:primitives|noise levels|seeds|cumulative stages)", domain)
    ]
    assert len(factors) == 4, (domain, factors)
    assert rai["ablation"]["total_solves"] == math.prod(factors), (
        f"total_solves == {rai['ablation']['total_solves']}, but the claim's domain "
        f"describes {factors} = {math.prod(factors)} solves"
    )
    inst = re.search(r"(K\s*=\s*\d+,\s*M\s*=\s*\d+,\s*n\s*=\s*\d+)", domain)
    assert inst, domain
    squeeze = lambda t: re.sub(r"\s+", "", t.lower())
    assert squeeze(inst.group(1)) in squeeze(rai["ablation"]["instance"]), (
        f"ablation.instance == {rai['ablation']['instance']!r} is not the instance the "
        f"claim's domain names, {inst.group(1)!r}"
    )
    # "0/504": none certified, out of a subset of the full design.
    good, through_b1 = (int(x) for x in rai["ablation"]["certified_through_B1"].split("/"))
    assert good == 0
    assert 0 < through_b1 < math.prod(factors)
    assert through_b1 % (math.prod(factors) // factors[-1]) == 0, (
        f"{through_b1} solves is not a whole number of the design's {factors[-1]} stages"
    )


# ======================================================================================
# M0-C06
# ======================================================================================

@pytest.mark.claim("M0-C06")
def test_rai_isotropic_weighting_is_structurally_forced():
    assert_claim_statement("M0-C06")      # eq["isotropic_weighting"], isotropy_required
    rai = CONVENTIONS["rai"]

    # The words "structurally forced" are decided by RUNNING the identity the table records.
    # The first version checked a textbook property of quadratic forms and never touched the
    # table (F-7.3).
    reason = rai["isotropy_reason"]
    m = re.search(r"(tau\*\|\|\w+\|\|\^2)\s*==\s*(tau\*\|\|R_i \w+\|\|\^2)", reason)
    assert m, f"isotropy_reason no longer records the identity it is cited for: {reason!r}"
    assert "SO(3)" in reason

    rng = np.random.default_rng(23)
    n = 10_000
    R = Rotation.random(n, rng=rng).as_matrix()
    r = rng.standard_normal((n, 3))
    tau = 2.7

    def weighted(side: str, W: np.ndarray) -> np.ndarray:
        """The recorded identity's side, with ||.||^2 read as the W-weighted square norm."""
        vec = np.einsum("nij,nj->ni", R, r) if "R_i" in side else r
        return tau * np.einsum("ni,ij,nj->n", vec, W, vec)

    lhs, rhs = m.group(1), m.group(2)
    assert ("R_i" in rhs) and ("R_i" not in lhs), reason
    iso = np.eye(3)
    base = weighted(lhs, iso)
    assert (np.abs(base - weighted(rhs, iso)) / np.abs(base)).max() < 1e-12

    M = rng.standard_normal((3, 3))
    W = M @ M.T + 3.0 * np.eye(3)                      # SPD, and not a multiple of I
    assert np.abs(W - np.trace(W) / 3.0 * np.eye(3)).max() > 1e-3
    rel = np.abs(weighted(lhs, W) - weighted(rhs, W)) / np.abs(weighted(lhs, W))
    assert rel.max() > 1e-3, "an anisotropic weight must break the recorded identity"
    assert rai["isotropy_required"] is True


# ======================================================================================
# M0-C07
# ======================================================================================

@pytest.mark.claim("M0-C07")
def test_delama_gal3_element_tangent_inverse_adjoint():
    assert_claim_statement("M0-C07")
    d = CONVENTIONS["delama"]

    # The four equation labels the claim names, each decided against delama's own LaTeX.
    cited = dict(
        re.findall(r'(element_order|tangent_order|inverse|adjoint)[^.]*?at eq "(\d+[a-z]?)"',
                   claim("M0-C07")["statement"])
    )
    assert d["eq"]["element"] == cited["element_order"]
    assert d["eq"]["tangent"] == cited["tangent_order"]
    assert d["eq"]["inverse"] == cited["inverse"]
    assert d["eq"]["adjoint"] == cited["adjoint"]

    rng = np.random.default_rng(56)
    A = Rotation.random(8, rng=rng).as_matrix()
    a = rng.standard_normal((8, 3))
    b = rng.standard_normal((8, 3))
    c = rng.standard_normal(8)

    slot_of = dict(zip(d["element_order"], d["element_slots"]))
    assert set(slot_of.values()) == {"rotation", "velocity", "position", "time"}
    X = np.stack([
        gal_from_rows(d["element_matrix"], {"A": A[k], "a": a[k], "b": b[k], "c": c[k]})
        for k in range(8)
    ])

    # The recorded inverse formula is EXECUTED (F-5: it used to be restated in the test body
    # and never read).  If any sign or slot in it is wrong, X @ X^-1 stops being I_5.
    body = d["inverse_formula"].split("=", 1)[1].strip()
    parts = [p.strip() for p in body.strip("()").split(",")]
    assert len(parts) == len(d["element_order"]), d["inverse_formula"]
    for k in range(8):
        env = {"A": A[k], "a": a[k], "b": b[k], "c": float(c[k])}
        vals = [run_formula(p, env, scalars=("c",)) for p in parts]
        inv = gal_from_rows(d["element_matrix"], dict(zip(d["element_order"], vals)))
        assert np.abs(X[k] @ inv - np.eye(5)).max() < 1e-12, (
            f"the recorded inverse_formula {d['inverse_formula']!r} is not an inverse"
        )
        assert np.abs(inv @ X[k] - np.eye(5)).max() < 1e-12

    # Which matrix slot is velocity and which is position is decided by the GROUP LAW, not
    # by the recorded word: only the position slot picks up the Galilean t2*v1 coupling.
    P = X[0] @ X[1]
    coupling = {}
    for slot, col in (("velocity", 3), ("position", 4)):
        simple = X[0][0:3, 0:3] @ X[1][0:3, col] + X[0][0:3, col]
        coupling[slot] = float(np.abs(P[0:3, col] - simple).max())
    assert coupling["velocity"] < 1e-12 and coupling["position"] > 1e-6, coupling
    index = {slot: i for i, slot in enumerate(d["element_slots"])}
    assert index["rotation"] == 0 and index["velocity"] == 1
    assert index["position"] == 2 and index["time"] == 3

    # The time coordinate is Ad-invariant: the last block row of Ad, computed from the
    # recorded element and the recorded tangent layout, must be the recorded row.
    blocks = _tangent_blocks(d["tangent_slots"])
    Ad = np.zeros((10, 10))
    for col_i in range(10):
        u = np.zeros(10)
        u[col_i] = 1.0
        uh = np.zeros((5, 5))
        uh[0:3, 0:3] = sk(u[blocks["rotation"]])
        uh[0:3, 3] = u[blocks["velocity"]]
        uh[0:3, 4] = u[blocks["position"]]
        uh[3, 4] = u[blocks["time"]][0]
        out = X[0] @ uh @ np.linalg.inv(X[0])
        v = np.zeros(10)
        v[blocks["rotation"]] = np.array([out[2, 1], out[0, 2], out[1, 0]])
        v[blocks["velocity"]] = out[0:3, 3]
        v[blocks["position"]] = out[0:3, 4]
        v[blocks["time"]] = out[3, 4]
        Ad[:, col_i] = v
    time_row = Ad[blocks["time"].start]
    measured = tuple(float(np.abs(time_row[blocks[s]]).max()) for s in d["tangent_slots"])
    assert d["adjoint_time_row"] == tuple(int(round(x)) for x in measured), (
        f"adjoint_time_row records {d['adjoint_time_row']}, Ad computes {measured}"
    )
    assert max(abs(x - round(x)) for x in measured) < 1e-9
    assert d["adjoint_time_invariant"] is (
        tuple(int(round(x)) for x in measured) == (0, 0, 0, 1)
    )

    # The letters differ between delama and gtsam; the ORDERING does not.
    assert d["tangent_slots"] == CONVENTIONS["gtsam"]["tangent_slots"]
    assert tuple(d["tangent_slots"]) == ("rotation", "velocity", "position", "time")

    # The constructor tuple is the matrix order, as the literature matrix records it.
    doc_ctor = re.search(r"tuple written `\(([^`]+)\)`", matrix_doc())
    assert doc_ctor, "the literature matrix no longer records delama's constructor tuple"
    doc_names = [re.sub(r"[^A-Za-z]", "", x).replace("k", "") for x in doc_ctor.group(1).split(",")]
    assert [n.lstrip("d") for n in d["ctor_order"]] == doc_names, (d["ctor_order"], doc_names)


# ======================================================================================
# M0-C08
# ======================================================================================

@pytest.mark.claim("M0-C08")
def test_delama_symmetry_group_dim20_and_right_action():
    assert_claim_statement("M0-C08")
    d = CONVENTIONS["delama"]

    # dim 20 == dim(Gal(3)) + dim(gal(3)), and the block shapes of Eqs. (36)/(37) must agree.
    assert sum(dim for _, dim in d["group_dim_components"]) == d["state_dim"]
    assert tuple(name for name, _ in d["group_dim_components"]) == ("Gal(3)", "gal(3)")
    assert sum(d["block_row_dims"]) == d["state_dim"]
    assert d["linearised_error_dim"] == d["state_dim"]
    assert d["A_hat_shape"] == d["B_hat_shape"] == (d["state_dim"], d["state_dim"])
    for label in (d["eq"]["error_dynamics_A"], d["eq"]["error_dynamics_B"]):
        assert_equation_says(
            "delama", label,
            require=[rf"\\mathbf\{{I\}}_\{{{d['block_row_dims'][0]}\}}",
                     rf"\\mathbf\{{0\}}_\{{{d['block_row_dims'][0]}"
                     rf"\\times {d['block_row_dims'][1]}\}}"],
        )

    # The semidirect laws are EXECUTED on the group they describe: (A, a) with A in Gal(3),
    # a in gal(3) and Ad_A[u] = A u A^-1.  X X^-1 must come out as the identity pair.
    rng = np.random.default_rng(1)

    def rand_alg():
        u = np.zeros((5, 5))
        u[0:3, 0:3] = sk(rng.standard_normal(3))
        u[0:3, 3] = rng.standard_normal(3)
        u[0:3, 4] = rng.standard_normal(3)
        u[3, 4] = float(rng.standard_normal())
        return u

    prod_parts = [p.strip() for p in
                  d["semidirect_product_law"].split("=", 1)[1].strip().strip("()").split(",")]
    inv_parts = [p.strip() for p in
                 d["semidirect_inverse_law"].split("=", 1)[1].strip().strip("()").split(",")]
    assert len(prod_parts) == len(inv_parts) == 2, (prod_parts, inv_parts)

    def Ad(M):
        Mi = np.linalg.inv(M)
        return lambda u: M @ u @ Mi

    def product(A1, a1, B1, b1):
        env = {"A": A1, "a": a1, "B": B1, "b": b1, "Ad_A": Ad(A1), "_inv": _inv}
        first = eval(mathify(prod_parts[0]), {"__builtins__": {}}, env)      # noqa: S307
        second = eval(prod_parts[1].replace("Ad_A[", "Ad_A(").replace("]", ")"),
                      {"__builtins__": {}}, env)                             # noqa: S307
        return first, second

    def inverse(A1, a1):
        env = {"A": A1, "a": a1, "Ad_Ainv": Ad(np.linalg.inv(A1)), "_inv": _inv}
        first = eval(inv_parts[0].replace("A^-1", "_inv(A)"),
                     {"__builtins__": {}}, env)                              # noqa: S307
        second = eval(inv_parts[1].replace("Ad_{A^-1}[a]", "Ad_Ainv(a)"),
                      {"__builtins__": {}}, env)                             # noqa: S307
        return first, second

    A, a = _rand_gal(rng), rand_alg()
    idA, ida = product(A, a, *inverse(A, a))
    assert np.abs(idA - np.eye(5)).max() < 1e-9, (
        f"the recorded semidirect laws do not compose to the identity: "
        f"{d['semidirect_product_law']!r} / {d['semidirect_inverse_law']!r}"
    )
    assert np.abs(ida).max() < 1e-9
    # ... and the product really is the semidirect one: it must not be the direct product.
    B, b = _rand_gal(rng), rand_alg()
    _, mixed = product(A, a, B, b)
    assert np.abs(mixed - (a + b)).max() > 1e-6

    # The action is a RIGHT action -- and the source itself says so, verbatim.
    html = source_html("delama")
    other = {"right": "left", "left": "right"}[d["action_handedness"]]
    assert html.count(f"{d['action_handedness']} group action") >= 1, (
        f"the delama source never says {d['action_handedness']!r} group action"
    )
    assert html.count(f"{other} group action") == 0, (
        f"the delama source also says {other!r} group action -- the handedness is not decided"
    )
    quoted = re.search(r"'([^']+)'", d["action_handedness_citation"])
    assert quoted, d["action_handedness_citation"]
    needle = re.sub(r"\s+", " ", quoted.group(1))
    tail = needle.split("free", 1)[-1].strip()
    assert tail.split()[0] == d["action_handedness"], needle
    assert f"Eq. ({d['eq']['state_action']})" in d["action_handedness_citation"]

    # The symmetry group is NOT the manifold.
    assert d["manifold"] == "Gal(3) x R^10"
    assert d["manifold"].split(" x ")[0] in d["group"]


# ======================================================================================
# M0-C09
# ======================================================================================

@pytest.mark.claim("M0-C09")
def test_delama_ten_dim_bias_with_four_virtual_channels():
    assert_claim_statement("M0-C09")
    d, g = CONVENTIONS["delama"], CONVENTIONS["gtsam"]

    # The virtual slices must be consistent with the recorded order and dimension: the two
    # virtual blocks are the TAIL of bias_order and together cover 4 of the 10 channels.
    slices = d["bias_virtual_slices"]
    assert all(isinstance(s, slice) for s in slices)
    for lo, hi in zip(slices, slices[1:]):
        assert lo.stop == hi.start, slices
    assert slices[-1].stop == d["bias_dim"]
    assert sum(s.stop - s.start for s in slices) == 4
    assert d["bias_virtual_names"] == tuple(d["bias_order"][-len(slices):]), (
        f"bias_virtual_names {d['bias_virtual_names']} are not the tail of bias_order "
        f"{d['bias_order']}"
    )
    assert d["bias_zero_process_noise"] == d["bias_virtual_names"]
    physical = slices[0].start
    assert physical == (len(d["bias_order"]) - len(slices)) * 3 == 6

    # The kinds and the virtual inputs are the source's own words, not ours.
    html = source_html("delama")
    assert len(d["bias_virtual_kinds"]) == len(slices)
    for kind, width in zip(d["bias_virtual_kinds"], (s.stop - s.start for s in slices)):
        assert kind.startswith("virtual "), kind
        assert kind.removeprefix("virtual ") in html, (
            f"the delama source never names a {kind!r}"
        )
    assert set(d["bias_virtual_inputs"]) == {
        n.split("_", 1)[1] + "_k" for n in d["bias_virtual_names"]
    }
    for name, width in zip(d["bias_virtual_names"], (s.stop - s.start for s in slices)):
        value = d["bias_virtual_inputs"][name.split("_", 1)[1] + "_k"]
        zero = re.fullmatch(r"0_\{(\d+)x(\d+)\}", value)
        if width == 1:
            assert value == "1", value
        else:
            assert zero and int(zero.group(1)) == width, (name, value, width)

    # The Q_d definition is an unnumbered "Define:" line of Algorithm 1.
    assert d["eq"]["bias_process_noise"] == COULD_NOT_EXTRACT

    # The two bias models cannot be silently interchanged.
    assert g["bias_dim"] == len(g["bias_order"]) * 3 == 6
    assert g["bias_virtual_slices"] == ()
    assert g["bias_zero_process_noise"] == ()


# ======================================================================================
# M0-C10
# ======================================================================================

#: The equation each row's estimator residual/error actually is.  Whether exp/log is
#: *essential* is then read off that equation's own LaTeX rather than taken on trust.
_RESIDUAL_EQ = {
    "rai": "rotation_residual",
    "barfoot": "cayley",
    "delama": "error_local_coords",
    "gtsam": "residual",
}
_TRANSCENDENTAL = r"\\log|\\exp|\\operatorname\{Log\}|\\operatorname\{Exp\}|logmap|\\ln"


@pytest.mark.claim("M0-C10")
def test_exp_log_essential_column():
    assert_claim_statement("M0-C10")      # the two delama eq labels
    expected = {k: v == "True" for k, v in
                re.findall(r'"(\w+)":\s*(True|False)', claim("M0-C10")["statement"])}
    assert set(expected) == set(CONVENTIONS), expected

    for key, want in expected.items():
        row = CONVENTIONS[key]
        assert row["exp_log_essential"] == want, (
            f"CONVENTIONS[{key!r}]['exp_log_essential'] == {row['exp_log_essential']}, "
            f"the claim statement says {want}"
        )
        # ... and the source's own residual/error equation must agree with the boolean.
        latex = equation_latex(key, row["eq"][_RESIDUAL_EQ[key]])
        found = bool(re.search(_TRANSCENDENTAL, latex))
        assert found == row["exp_log_essential"], (
            f"{key} Eq. ({row['eq'][_RESIDUAL_EQ[key]]}) "
            f"{'contains' if found else 'does not contain'} a transcendental map, but "
            f"exp_log_essential is recorded as {row['exp_log_essential']}"
        )

    d = CONVENTIONS["delama"]
    for label, entry in d["exp_log_dependencies"].items():
        assert entry["eq"] == d["eq"][label]
        latex = equation_latex("delama", entry["eq"])
        assert set(entry["requires"]) == {"log", "J_L^{-1}"}, entry["requires"]
        assert re.search(r"\\log", latex), label
        assert re.search(r"\\mathbf\{J\}_\{\\mathrm\{L\}\}", latex), label
        assert entry["requires_eq"]["log"] == d["eq"]["log"]
        assert entry["requires_eq"]["J_L^{-1}"] == d["eq"]["left_jacobian"]

    # Falsifiability anchor for the two False entries: rai removes Log via the chordal
    # residual AND Exp via Cayley.  Removing only one leaves a non-polynomial problem.
    rai = CONVENTIONS["rai"]
    assert_equation_says("rai", rai["eq"]["chordal_residual"],
                         require=[r"\\mathrm\{Exp\}", r"_\{F\}\^\{2\}"])
    assert_equation_says("rai", rai["eq"]["cayley"], require=[r"\\mathrm\{Cay\}"])


# ======================================================================================
# M0-C11
# ======================================================================================

@pytest.mark.claim("M0-C11")
def test_error_handedness_and_bias_correction_side_disagree_on_both_axes():
    assert_claim_statement("M0-C11")
    d, g = CONVENTIONS["delama"], CONVENTIONS["gtsam"]

    # -- the ERROR handedness is decided by testing the invariance the table names ---------
    rng = np.random.default_rng(34)
    errors = {
        "delama": lambda X, Xh: X @ np.linalg.inv(Xh),       # Upsilon Upsilon_hat^-1, eq 34
        "gtsam": lambda X, Xh: np.linalg.inv(Xh) @ X,        # xi_hat^-1 xi, eq 21
    }
    for key, err in errors.items():
        row = CONVENTIONS[key]
        kind = row["error_handedness"]
        assert kind.endswith("-invariant"), kind
        side = kind.split("-")[0]
        X, Xh, S = _rand_gal(rng), _rand_gal(rng), _rand_gal(rng)
        base = err(X, Xh)
        translated = {
            "left": err(S @ X, S @ Xh),
            "right": err(X @ S, Xh @ S),
        }
        assert np.abs(translated[side] - base).max() < 1e-9, (
            f"{key} records a {kind} error, but translating on the {side} changes it"
        )
        other = "left" if side == "right" else "right"
        assert np.abs(translated[other] - base).max() > 1e-6
    assert d["error_handedness"] != g["error_handedness"]

    # -- the bias-correction SIDE is read off the cited equation's own LaTeX --------------
    for key, hat in (("delama", r"\\hat\{\\mathbf\{\\Upsilon\}\}"), ("gtsam", r"\\hat\\Upsilon")):
        row = CONVENTIONS[key]
        latex = equation_latex(key, row["eq"]["bias_correction"])
        rhs = latex.split("\\approx", 1)[-1].split("\\simeq", 1)[-1]
        e = re.search(r"\\exp|\\operatorname\{Exp\}", rhs)
        h = re.search(hat, rhs)
        assert e and h, rhs[:200]
        side = "left" if e.start() < h.start() else "right"
        assert row["bias_correction_side"] == side, (
            f"{key} Eq. ({row['eq']['bias_correction']}) applies the correction on the "
            f"{side}; the table records {row['bias_correction_side']!r}"
        )
    assert d["bias_correction_side"] != g["bias_correction_side"]

    # Eq-label collision guard.  The label "34" appears in BOTH sources meaning entirely
    # different equations.  The first version asserted `"a" != "b"` on two string literals
    # written in this file (F-7.1); the labels are now resolved in their own sources and the
    # equations found there must differ.
    assert d["eq"]["error_local_coords"] == g["eq"]["bias_correction"]
    assert equation_latex("delama", d["eq"]["error_local_coords"]) != equation_latex(
        "gtsam", g["eq"]["bias_correction"]
    )

    # The right perturbation that gtsam's left-invariant error is equivalent to, checked
    # against the installed library rather than recorded as a sentence.
    retr = g["retraction"]
    m = re.search(r"X\s*\(\+\)\s*delta\s*==\s*(.+?)\s*\(", retr)
    assert m, retr
    rhs = m.group(1).replace(" ", "")
    assert rhs in ("XExp(delta)", "Exp(delta)X"), retr
    X = gtsam.Gal3(gtsam.Rot3.Expmap(np.array([0.1, 0.2, 0.3])), np.array([1.0, 2.0, 3.0]),
                   np.array([0.4, 0.5, 0.6]), 0.7)
    delta = np.arange(10) * 0.001
    E = gtsam.Gal3.Expmap(delta)
    built = X.compose(E) if rhs == "XExp(delta)" else E.compose(X)
    assert np.abs(X.retract(delta).matrix() - built.matrix()).max() < 1e-12, (
        f"the recorded retraction {retr!r} is not what gtsam {_installed_gtsam_version()} does"
    )
    assert ("right" in retr) == (rhs == "XExp(delta)")

    # The ERROR handedness and the state ACTION handedness are different statements.
    assert d["action_handedness"] == "right"


# ======================================================================================
# M0-C12
# ======================================================================================

def rank_verdict(criterion: dict, M: np.ndarray) -> tuple[float, bool]:
    """Run a recorded numerical-rank rule.  Every ingredient comes out of the table."""
    spectrum = criterion["spectrum"]
    if "singular value" in spectrum:
        vals = np.linalg.svd(M, compute_uv=False)
    elif "eigenvalue" in spectrum:
        vals = np.sort(np.abs(np.linalg.eigvalsh(M)))[::-1]
    else:
        raise AssertionError(f"unknown recorded spectrum {spectrum!r}")

    stat, floor = criterion["statistic"], criterion["relative_floor"]
    kind = criterion["kind"]
    if stat.startswith("#{") and "lambda_max" in stat:
        assert f"{floor:g}" in stat.replace("1e-06", "1e-6"), (
            f"the recorded statistic {stat!r} does not spell its own floor {floor!r}"
        )
        value = float((vals > floor * vals.max()).sum())
        assert "count" in kind and "eigenvalue" in kind, kind
    elif stat.startswith("log10(") and "/" in stat:
        num, den = (p.strip() for p in stat[len("log10("):-1].split("/"))
        order = {"sigma_1": 0, "sigma_2": 1}
        assert num in order and den in order and num != den, stat
        value = float(np.log10(vals[order[num]] / vals[order[den]]))
        assert math.isclose(floor, 10.0 ** -criterion["threshold"], rel_tol=1e-12), (
            f"a log-ratio rule with threshold {criterion['threshold']} has relative floor "
            f"{10.0 ** -criterion['threshold']:g}, not {floor!r}"
        )
        assert "ratio" in kind and "singular" in kind, kind
    else:
        raise AssertionError(f"unknown recorded statistic {stat!r}")

    rel = {"==": operator.eq, ">=": operator.ge, ">": operator.gt,
           "<=": operator.le, "<": operator.lt}
    assert criterion["relation"] in rel, criterion["relation"]
    return value, bool(rel[criterion["relation"]](value, criterion["threshold"]))


@pytest.mark.claim("M0-C12")
def test_barfoot_log_svr_and_rai_eigenvalue_rules_are_distinct():
    b, r = CONVENTIONS["barfoot"], CONVENTIONS["rai"]
    bc, rc = b["rank1_criterion"], r["rank1_criterion"]

    # Both criteria are prose, so both eq entries must be the sentinel.
    assert b["eq"]["rank1_criterion"] == COULD_NOT_EXTRACT
    assert b["eq"]["sparsity"] == COULD_NOT_EXTRACT
    assert r["eq"]["rank1_criterion"] == COULD_NOT_EXTRACT

    # The claim statement fixes barfoot's statistic, relation and threshold.
    stmt = claim("M0-C12")["statement"]
    thr = re.search(r"(log10\(sigma_1 / sigma_2\))\s*(>=|>)\s*(\d+)", stmt)
    assert thr, stmt
    assert bc["statistic"] == thr.group(1)
    assert bc["relation"] == thr.group(2)
    assert bc["threshold"] == int(thr.group(3))
    assert "measurement sparsity" in b["tightness_degrades_with"]
    assert rc != bc, "the two numerical-rank tests must not be aliased to one another"

    # Numerical anchor, with the disagreement in the direction the claim's test_hint states:
    # a spectrum whose second eigenvalue sits between the two recorded relative floors.
    assert rc["relative_floor"] < bc["relative_floor"], (rc["relative_floor"], bc["relative_floor"])
    gap = math.sqrt(rc["relative_floor"] * bc["relative_floor"])
    rng = np.random.default_rng(12)
    Q, _ = np.linalg.qr(rng.standard_normal((10, 10)))

    def sym(eigs):
        Z = Q @ np.diag(np.asarray(eigs)) @ Q.T
        return (Z + Z.T) / 2

    Z = sym([1.0, gap] + [1e-9] * 8)
    b_value, b_says = rank_verdict(bc, Z)
    r_value, r_says = rank_verdict(rc, Z)
    assert b_says is True, f"barfoot's recorded rule rejects a log-SVR of {b_value}"
    assert r_says is False, f"rai's recorded rule accepts numerical rank {r_value}"
    assert b_says != r_says

    # ... and the two rules must still agree on the unambiguous spectra.
    rank1 = sym([1.0] + [1e-12] * 9)
    rank2 = sym([1.0, 0.5] + [1e-12] * 8)
    assert rank_verdict(bc, rank1)[1] is True
    assert rank_verdict(rc, rank1)[1] is True
    assert rank_verdict(bc, rank2)[1] is False
    assert rank_verdict(rc, rank2)[1] is False

    assert_equation_says("barfoot", b["eq"]["shor_relaxation"],
                         require=[r"\\succeq 0"], forbid=[r"\\mbox\{rank\}"])
    assert_equation_says("barfoot", b["eq"]["rank1_sdp"],
                         require=[r"\\mbox\{rank\}\(\\mathbf\{X\}\)=1"])
    assert b["relaxation"].startswith(f"Shor Eq. ({b['eq']['shor_relaxation']})")


# ======================================================================================
# M0-C13
# ======================================================================================

def _ordinal(word: str) -> int:
    return {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4,
            "first": 1, "second": 2, "third": 3, "fourth": 4}[word.lower()]


@pytest.mark.claim("M0-C13")
def test_seven_ordering_incompatibilities_against_installed_gtsam():
    stmt = claim("M0-C13")["statement"]
    assert set(INCOMPATIBILITIES) == set(re.findall(r"\bO[1-7]\b", stmt)) == {
        f"O{i}" for i in range(1, 8)
    }
    for key, row in INCOMPATIBILITIES.items():
        assert row["a"]["source"] in CONVENTIONS and row["b"]["source"] in CONVENTIONS, key
        assert row["blocks"], key
        for side in ("a", "b"):
            assert row[side]["source"] in row["where"], (
                f"{key}.where does not name the {side}-side source it is about"
            )

    assert _installed_gtsam_version() == CONVENTIONS["gtsam"]["package_version"]
    g = CONVENTIONS["gtsam"]

    # -- O1: matrix column order vs constructor argument order ---------------------------
    o1 = INCOMPATIBILITIES["O1"]
    ctor_slot = dict(zip(g["ctor_order"], g["ctor_slots"]))
    assert set(ctor_slot.values()) == {"rotation", "position", "velocity", "time"}
    args = {"rotation": gtsam.Rot3.Identity(), "position": np.array([1.0, 2.0, 3.0]),
            "velocity": np.array([10.0, 20.0, 30.0]), "time": 7.0}
    X = gtsam.Gal3(*[args[ctor_slot[name]] for name in g["ctor_order"]])
    M = X.matrix()
    el_slot = dict(zip(g["element_order"], g["element_slots"]))
    col_of = {el_slot[name]: 3 + i for i, name in enumerate(g["element_order"][1:3])}
    assert set(col_of) == {"velocity", "position"}, g["element_slots"]
    for slot, col in col_of.items():
        assert np.allclose(M[0:3, col], args[slot]), (
            f"element_order {g['element_order']} puts {slot} in column {col}; gtsam does not"
        )
    assert M[3, 4] == args["time"]

    # The two `says` entries are parsed and RUN, not substring-matched (F-4).
    for slot in ("velocity", "position"):
        m = re.search(rf"{slot} in (?:matrix )?column index (\d+)", o1["a"]["says"])
        assert m, o1["a"]["says"]
        assert np.allclose(M[0:3, int(m.group(1))], args[slot]), (
            f"O1.a.says puts {slot} in column {m.group(1)}; gtsam disagrees"
        )
    sig = re.search(r"Gal3\(([^)]*)\)", o1["b"]["says"])
    assert sig and tuple(x.strip() for x in sig.group(1).split(",")) == g["ctor_order"], (
        o1["b"]["says"]
    )
    for slot, word in re.findall(r"(\w+) (?:is )?the (\w+) argument", o1["b"]["says"]):
        name = next(n for n, s in ctor_slot.items() if s == slot)
        assert g["ctor_order"].index(name) + 1 == _ordinal(word), o1["b"]["says"]
    assert o1["b"]["eq"] == COULD_NOT_EXTRACT     # an API fact with no numbered equation
    assert set(o1["blocks"]) == {g["element_order"][1], g["element_order"][2]}
    rt = o1["runtime_check"]
    m = re.search(r"Gal3\(([^)]*)\)\.matrix\(\) puts \[([^]]*)\] in column index (\d+) and "
                  r"\[([^]]*)\] in column index (\d+)", rt)
    assert m, rt
    probe = []
    for token in [t.strip() for t in m.group(1).split("], [")]:
        token = token.strip("[]")
        if "Identity" in token:
            probe.append(gtsam.Rot3.Identity())
        elif "," in token:
            probe.append(np.array([float(x) for x in token.split(",")]))
        else:
            probe.append(float(token))
    PM = gtsam.Gal3(*probe).matrix()
    for vec, col in ((m.group(2), m.group(3)), (m.group(4), m.group(5))):
        assert np.allclose(PM[0:3, int(col)], [float(x) for x in vec.split(",")]), (
            f"O1.runtime_check is not what gtsam {_installed_gtsam_version()} does"
        )

    # -- O2: Gal3 tangent vs NavState tangent, and the P_N permutation -------------------
    o2 = INCOMPATIBILITIES["O2"]
    blocks = _tangent_blocks(g["tangent_slots"])
    ns_blocks, at = {}, 0
    for slot in g["navstate_tangent_slots"]:
        ns_blocks[slot] = slice(at, at + 3)
        at += 3
    assert at == 9
    for i in range(blocks["velocity"].start, blocks["velocity"].stop):
        e10 = np.zeros(10)
        e10[i] = 1.0
        hat = gtsam.Gal3.Hat(e10)
        assert np.linalg.norm(hat[:, col_of["velocity"]]) > 0.5
        assert np.linalg.norm(hat[:, col_of["position"]]) < 1e-12

        # the SAME index moves POSITION in a NavState -- that is the incompatibility
        assert ns_blocks["position"].start <= i < ns_blocks["position"].stop
        e9 = np.zeros(9)
        e9[i] = 1.0
        moved = gtsam.NavState().retract(e9)
        assert np.linalg.norm(moved.position()) > 0.5
        assert np.linalg.norm(moved.velocity()) < 1e-12

    rows = o2["P_N_rows"]
    assert g["navstate_row_selection"] == rows
    assert tuple(sorted(rows)) == tuple(range(9))
    # P_N_blocks says the same thing in slice notation; it is parsed, not trusted.
    spans = re.findall(r"(\d+):(\d+)", o2["P_N_blocks"])
    assert spans, o2["P_N_blocks"]
    assert tuple(i for lo, hi in spans for i in range(int(lo), int(hi))) == rows, (
        f"P_N_blocks {o2['P_N_blocks']!r} does not expand to P_N_rows {rows}"
    )
    assert g["navstate_block_map"] == tuple(int(lo) for lo, _ in spans)
    assert tuple(tuple(rows[ns_blocks[s]]) for s in g["navstate_tangent_slots"]) == tuple(
        tuple(range(blocks[s].start, blocks[s].stop)) for s in g["navstate_tangent_slots"]
    ), "P_N does not map each NavState block onto its own Gal3 block"
    assert g["navstate_block_map_symbol"] == COULD_NOT_EXTRACT

    rng = np.random.default_rng(38)
    eps = 1e-6
    Xg = gtsam.Gal3(gtsam.Rot3.Expmap(np.array([0.2, -0.1, 0.4])), np.array([1.0, 2.0, 3.0]),
                    np.array([0.4, -0.5, 0.6]), 0.0)
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

    for side, which in (("a", g["tangent_order"]), ("b", g["navstate_tangent_order"])):
        said = re.search(r"\(([^)]*)\)", o2[side]["says"])
        assert said and tuple(x.strip() for x in said.group(1).split(",")) == which, (
            o2[side]["says"]
        )

    # -- O3: imuBias::ConstantBias is (accel, gyro) --------------------------------------
    o3 = INCOMPATIBILITIES["O3"]
    probe = {"b_a": np.array([1.0, 2.0, 3.0]), "b_g": np.array([40.0, 50.0, 60.0])}
    probe["b_omega"] = probe["b_g"]
    cb = gtsam.imuBias.ConstantBias(probe["b_a"], probe["b_g"])
    assert np.allclose(cb.accelerometer(), probe["b_a"])
    assert np.allclose(cb.gyroscope(), probe["b_g"])
    assert np.allclose(cb.vector(), np.concatenate([probe[n] for n in g["bias_order"]])), (
        f"gtsam bias_order {g['bias_order']} is not the order vector() returns"
    )
    assert g["bias_order"] != CONVENTIONS["rai"]["bias_order"]
    for side, row_key in (("a", "gtsam"), ("b", "rai")):
        said = re.search(r"\[([^]]*)\]", o3[side]["says"])
        assert said, o3[side]["says"]
        assert tuple(x.strip() for x in said.group(1).split(";")) == CONVENTIONS[row_key]["bias_order"], (
            f"O3.{side}.says records a different bias order than CONVENTIONS[{row_key!r}]"
        )
    said_delama = re.search(r"delama uses \(([^)]*)\)", o3["b"]["says"])
    assert said_delama, o3["b"]["says"]
    assert tuple(x.strip() for x in said_delama.group(1).split(",")) == (
        CONVENTIONS["delama"]["bias_order"]
    )
    assert o3["b"]["eq"] == COULD_NOT_EXTRACT
    m = re.search(r"ConstantBias\(\[([^]]*)\],\[([^]]*)\]\)\.vector\(\)\s*==\s*\[([^]]*)\]",
                  o3["runtime_check"])
    assert m, o3["runtime_check"]
    acc = np.array([float(x) for x in m.group(1).split(",")])
    gyr = np.array([float(x) for x in m.group(2).split(",")])
    out = np.array([float(x) for x in m.group(3).split(",")])
    assert np.allclose(gtsam.imuBias.ConstantBias(acc, gyr).vector(), out), (
        f"O3.runtime_check is not what gtsam {_installed_gtsam_version()} does"
    )

    # -- O4: integrateMeasurement takes (acc, omega) -------------------------------------
    o4 = INCOMPATIBILITIES["O4"]
    params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    zero = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    order = g["imu_sample_order"]
    sample = {"acc": np.array([1.0, 2.0, 3.0]), "omega": np.zeros(3)}
    pim = gtsam.PreintegratedImuMeasurementsG(params, zero)
    pim.integrateMeasurement(*[sample[k] for k in order], 0.1)
    assert np.abs(pim.deltaRij().matrix() - np.eye(3)).max() < 1e-12
    assert np.allclose(pim.deltaVij(), sample["acc"] * 0.1)
    swapped = gtsam.PreintegratedImuMeasurementsG(params, zero)
    swapped.integrateMeasurement(*[sample[k] for k in order[::-1]], 0.1)
    assert np.abs(swapped.deltaRij().matrix() - np.eye(3)).max() > 1e-3
    assert np.linalg.norm(swapped.deltaVij()) < 1e-12
    said = re.search(r"u = \[([^]]*)\]", o4["a"]["says"])
    assert said, o4["a"]["says"]
    assert tuple(x.strip()[0] for x in said.group(1).split(";")) == tuple(s[0] for s in order)
    # rai lists the two channels the other way round -- and rai's own equations say so.
    pair = re.search(r"\((\d+[a-z])\)\((\d+[a-z])\)\s*list\s*(\w+)\s*then\s*(\w+)",
                     o4["b"]["says"])
    assert pair, o4["b"]["says"]
    assert o4["b"]["eq"] == pair.group(1), (o4["b"]["eq"], pair.group(1))
    greek = {"omega_tilde": r"\\tilde\{\\boldsymbol\{\\omega\}\}",
             "a_tilde": r"\\tilde\{\\mathbf\{a\}\}"}
    assert_equation_says("rai", pair.group(1), require=[greek[pair.group(3)]])
    assert_equation_says("rai", pair.group(2), require=[greek[pair.group(4)]])
    assert pair.group(3).split("_")[0] != order[0] and pair.group(4).split("_")[0] == "a"
    m = re.search(r"deltaVij\(\) == \[([^]]*)\] at dt = ([\d.]+)", o4["runtime_check"])
    assert m, o4["runtime_check"]
    dtv = np.array([float(x) for x in m.group(1).split(",")])
    dt4 = float(m.group(2))
    probe_pim = gtsam.PreintegratedImuMeasurementsG(params, zero)
    probe_pim.integrateMeasurement(dtv / dt4, np.zeros(3), dt4)
    assert np.allclose(probe_pim.deltaVij(), dtv), (
        f"O4.runtime_check is not what gtsam {_installed_gtsam_version()} does"
    )

    # -- O5: barfoot SE(3) tangent is [rho; phi], translation first ----------------------
    o5 = INCOMPATIBILITIES["O5"]
    bt = CONVENTIONS["barfoot"]
    said = re.search(r"xi = \[([^]]*)\]", o5["a"]["says"])
    assert said and tuple(x.strip() for x in said.group(1).split(";")) == bt["tangent_order"]
    assert o5["blocks"] == bt["tangent_order"]
    lead = re.search(r"--\s*(\w+) first", o5["a"]["says"])
    assert lead and lead.group(1) == "translation", o5["a"]["says"]
    assert bt["tangent_slots"][0] == "position"
    said_b = re.search(r"(.+?) put (\w+) first", o5["b"]["says"])
    assert said_b, o5["b"]["says"]
    named = [k for k in re.findall(r"\w+", said_b.group(1)) if k in CONVENTIONS]
    assert named, o5["b"]["says"]
    for key in named:
        assert CONVENTIONS[key]["tangent_slots"][0] == said_b.group(2)
    assert bt["tangent_slots"][0] != said_b.group(2)

    # -- O6: the letters r / rho ---------------------------------------------------------
    o6 = INCOMPATIBILITIES["O6"]
    dl = CONVENTIONS["delama"]
    assert g["ctor_order"][g["ctor_slots"].index("position")] == o6["blocks"][0]
    assert bt["tangent_order"][bt["tangent_slots"].index("position")] == o6["blocks"][1]
    assert dl["tangent_order"][dl["tangent_slots"].index("position")] == o6["blocks"][0]
    assert dl["tangent_slots"] == g["tangent_slots"]
    said6 = re.search(r"delama's \(([^)]*)\) IS gtsam's \(([^)]*)\)", o6["b"]["says"])
    assert said6, o6["b"]["says"]
    assert tuple(x.strip() for x in said6.group(1).split(",")) == dl["tangent_order"]
    assert tuple(x.strip() for x in said6.group(2).split(",")) == g["tangent_order"]
    assert dl["tangent_order"] != g["tangent_order"]

    # -- O7: gravity position-block sign, checked by INTEGRATING both routes --------------
    o7 = INCOMPATIBILITIES["O7"]
    assert o7["a"]["source"] == "delama" and o7["b"]["source"] == "gtsam"
    ca, cb_ = o7["a"]["gravity_position_coefficient"], o7["b"]["gravity_position_coefficient"]
    assert ca != cb_
    assert o7["blocks"] == ("p",)

    rng = np.random.default_rng(718)
    for _ in range(50):
        dt = float(rng.uniform(0.2, 1.5))
        gv = np.array([0.0, 0.0, -9.81])
        Ri = Rotation.random(1, rng=rng).as_matrix()[0]
        vi, pi = rng.standard_normal(3), rng.standard_normal(3)
        dR = Rotation.random(1, rng=rng).as_matrix()[0]
        dv, dp = rng.standard_normal(3), rng.standard_normal(3)

        # the Gal(3) route, with the left factor the table records
        Gamma = gal_from_rows(o7["a"]["says"], {"g": gv, "dt": dt})
        assert np.allclose(Gamma[0:3, 4], ca * gv * dt**2), (
            f"O7.a.says and gravity_position_coefficient {ca} disagree"
        )
        Xj = Gamma @ gal(Ri, vi, pi, 0.0) @ gal(dR, dv, dp, dt)

        # the SE_2(3)/NavState route, with gtsam's own coefficient
        pj = cb_ * gv * dt**2 + (pi + vi * dt) + Ri @ dp
        vj = gv * dt + vi + Ri @ dv
        assert np.abs(Xj[0:3, 4] - pj).max() < 1e-9, (
            "the two recorded gravity coefficients do not describe the same trajectory"
        )
        assert np.abs(Xj[0:3, 3] - vj).max() < 1e-9

        # ... and the reason's arithmetic: using gtsam's coefficient inside the Gal(3) left
        # factor is off by exactly +g*dt^2, the t2*v1 coupling the reason names.
        wrong = Gamma.copy()
        wrong[0:3, 4] = cb_ * gv * dt**2
        assert np.abs((wrong @ gal(Ri, vi, pi, 0.0) @ gal(dR, dv, dp, dt))[0:3, 4]
                      - (pj + gv * dt**2)).max() < 1e-9

    reason = o7["reason"]
    assert "not a contradiction" in reason
    coupling = re.search(r"Galilean (\S+) coupling term of the \w+ group product, eq (\d+[a-z]?)",
                         reason)
    assert coupling, reason
    assert coupling.group(2) == g["eq"]["product"], (
        f"the O7 reason cites eq {coupling.group(2)} for the group product; the table "
        f"records {g['eq']['product']}"
    )
    assert_equation_says("gtsam", coupling.group(2),
                         require=[coupling.group(1).replace("t2*v1", r"t_2v_1")])


# ======================================================================================
# M0-C14
# ======================================================================================

@pytest.mark.claim("M0-C14")
def test_gtsam_endpoint_relation_is_W_phi_X_U_and_not_H_X_D():
    assert_claim_statement("M0-C14")      # eq["endpoint"] == "17"
    g = CONVENTIONS["gtsam"]

    # The regime is pinned by the claim's domain field, not by this file: err_wrong is
    # exactly ||v_i|| * dt_ij, so both have to be stated for the > 1e-2 bar to mean anything
    # (verification finding F-1).
    want_dt = claim_number("M0-C14", "domain", r"dt_ij\s*~\s*([\d.]+)")
    want_v = claim_number("M0-C14", "domain", r"\|\|v_i\|\|\s*>\s*([\d.]+)")
    bar_ok = claim_number("M0-C14", "statement", r"to <= (1e-\d+)")
    bar_wrong = claim_number("M0-C14", "statement", r"more than (1e-\d+)")

    params, bias, pim = _pim_with_motion(n=50, dt=want_dt / 50)
    dt_ij = pim.deltaTij()
    assert dt_ij == pytest.approx(want_dt, abs=1e-9)
    g_vec = params.n_gravity
    assert np.allclose(g_vec, [0.0, 0.0, -9.81])

    X_i = gtsam.NavState(
        gtsam.Rot3.Expmap(np.array([0.11, -0.23, 0.31])),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, -0.4, 0.3]),
    )
    assert np.linalg.norm(X_i.velocity()) > want_v
    X_j = pim.predict(X_i, bias)

    # The three endpoint factors are BUILT from the strings the table records.
    env = {
        "NavState": gtsam.NavState,
        "I": gtsam.Rot3.Identity(),
        "g": np.asarray(g_vec, dtype=float),
        "dt_ij": float(dt_ij),
        "dt": float(dt_ij),
        "dR_ij": pim.deltaRij(),
        "dp_ij": pim.deltaPij(),
        "dv_ij": pim.deltaVij(),
        "R": X_i.attitude(),
        "p": X_i.position(),
        "v": X_i.velocity(),
    }
    factors = g["endpoint_factors"]
    W = eval(factors["W_ij"], {"__builtins__": {}}, env)                   # noqa: S307
    U = eval(factors["U_ij"], {"__builtins__": {}}, env)                   # noqa: S307
    phi = gtsam.NavState(
        *eval(factors["phi_dt"].split("=", 1)[1].strip(), {"__builtins__": {}}, env)
    )                                                                      # noqa: S307
    assert isinstance(W, gtsam.NavState) and isinstance(U, gtsam.NavState)

    # ... and composed in the order the recorded endpoint_relation gives.
    lhs, rhs = (s.strip() for s in g["endpoint_relation"].split("=", 1))
    assert lhs == "X_j", g["endpoint_relation"]
    terms = [t.strip() for t in rhs.split(" . ")]
    built = {"W_ij": W, "U_ij(beta_i)": U, "phi_{dt_ij}(X_i)": phi}
    assert set(terms) == set(built), g["endpoint_relation"]
    correct = built[terms[0]]
    for t in terms[1:]:
        correct = correct.compose(built[t])
    err_correct = np.linalg.norm(X_j.localCoordinates(correct))
    assert err_correct <= bar_ok, (
        f"{g['endpoint_relation']} does not reproduce predict(): {err_correct:.3e}"
    )

    # ... and the coasting-free, H X D-shaped variant must be visibly WRONG.  Without this
    # half a trivially-equal U would pass the test.
    assert "coasting map dropped" in g["endpoint_rejected_form"]
    err_wrong = np.linalg.norm(X_j.localCoordinates(W.compose(X_i).compose(U)))
    assert err_wrong > bar_wrong, (
        f"W . X_i . U is not distinguishable from the truth: {err_wrong:.3e}"
    )
    # The margin is exactly ||v_i|| * dt_ij, which is why the claim's domain pins both.
    assert err_wrong == pytest.approx(np.linalg.norm(X_i.velocity()) * dt_ij, rel=1e-6)
    assert want_v * want_dt > bar_wrong, (
        "the regime the claim's domain states does not itself clear the bar"
    )


# ======================================================================================
# M0-C15
# ======================================================================================

@pytest.mark.claim("M0-C15")
def test_delama_eq18_gamma_equals_T_minus_dt_times_Hprime_exactly():
    """Delama Eq. (18) IS the pre-flight F2 corrected left factor -- bit-identically.

    Exactness (not a tolerance) is expected because both expressions are built from the same
    literals.  The reversed product is checked too, because a test of the first half alone
    would pass for any commuting pair and would say nothing about side placement.

    Verification finding F-2: the reversed error is exactly ``||g|| dt^2``, so the claim's
    ``> 1e-6`` bar holds only above a dt floor.  That floor is now stated in the claim's
    domain and enforced here, and the exact scaling law is asserted alongside the bar so the
    property is checked scale-free as well.
    """
    o7a = INCOMPATIBILITIES["O7"]["a"]
    assert o7a["eq"] == CONVENTIONS["delama"]["eq"]["gravity_left_factor"]

    # H' and T(-dt) are the forms the CLAIM writes out; Gamma is the matrix the TABLE records
    # for Delama Eq. (18).  Neither side is restated in this file.
    stmt = claim("M0-C15")["statement"]
    hp = re.search(r"H' = (\[\[.*?\]\])", stmt)
    tt = re.search(r"T\(-dt\) = (\[\[.*?\]\])", stmt)
    assert hp and tt, stmt
    dt_floor = claim_number("M0-C15", "domain", r"\|dt\|\s*>=\s*([\d.e-]+)")
    g_floor = claim_number("M0-C15", "domain", r"\|\|g\|\|\s*>=\s*([\d.e-]+)")
    bar = claim_number("M0-C15", "statement", r"Hp @ T\) > (1e-\d+)")

    rng = np.random.default_rng(1815)
    for _ in range(200):
        dt = float(rng.uniform(dt_floor, 2.0)) * rng.choice([-1.0, 1.0])
        gv = rng.standard_normal(3)
        gv *= rng.uniform(g_floor, 20.0) / np.linalg.norm(gv)

        Gamma = gal_from_rows(o7a["says"], {"g": gv, "dt": dt})
        Hp = gal_from_rows(hp.group(1), {"g": gv, "dt": dt})
        T = gal_from_rows(tt.group(1), {"g": gv, "dt": dt})

        assert np.array_equal(Gamma, T @ Hp), (
            f"the recorded Eq. (18) factor is not T(-dt) @ H' at dt={dt:g}"
        )
        assert np.linalg.norm(Gamma - T @ Hp) == 0.0

        # Side placement is load-bearing: on the right, T corrupts the position block by
        # exactly ||g|| dt^2.
        reversed_err = np.linalg.norm(Gamma - Hp @ T)
        assert reversed_err == pytest.approx(np.linalg.norm(gv) * dt**2, rel=1e-12)
        assert reversed_err > bar, (
            f"the reversed product is indistinguishable at dt={dt:g}, "
            f"||g||={np.linalg.norm(gv):g}"
        )
    # The bar is only meaningful because the domain pins dt: at the 10 kHz sample interval
    # the domain excludes, the very same expression falls under it.
    assert g_floor * dt_floor**2 > bar > 9.81 * 1e-4**2


# ======================================================================================
# Equation labels and prose, row by row, against the sources they cite
# ======================================================================================

#: ``path -> (patterns the cited equation must contain, patterns it must not)``.
#: The patterns describe what the equation *says*; the label looked up is the one the table
#: records, so a corrupted label lands on an equation that fails its signature.
EQ_SIGNATURES: dict[tuple[str, ...], tuple[tuple[str, ...], tuple[str, ...]]] = {
    ("rai", "eq", "cayley"): ((r"\\mathrm\{Cay\}", r"2\\tan\(\\theta/2\)"), ()),
    ("rai", "eq", "cayley_right_jacobian"): ((r"\\mathbf\{J\}_\{\\mathrm\{Cay\}\}",), ()),
    ("rai", "eq", "qcqp"): ((r"f_\{\\mathrm\{QCQP\}\}", r"\\mathbf\{A\}_\{j\}"), ()),
    ("rai", "eq", "sdp"): ((r"f_\{\\mathrm\{SDP\}\}", r"\\mathbb\{S\}\^\{n\}_\{\+\}"), ()),
    ("rai", "eq", "chordal_residual"): ((r"\\mathrm\{Exp\}", r"_\{F\}\^\{2\}"), ()),
    ("rai", "eq", "xi_definition"): ((r"\\boldsymbol\{\\xi\}_\{ij\}\\triangleq",), ()),
    ("rai", "eq", "cayley_determinant"): ((r"\\det", r"1\+\\tfrac\{1\}\{4\}"), ()),
    ("rai", "eq", "denominator_clearing"): (
        (r"\\mathbf\{R\}_\{j\}\\bigl\(\\mathbf\{I\}_\{3\}-\\tfrac\{1\}\{2\}",), ()),
    ("rai", "eq", "rotation_residual"): ((r"\\mathbf\{r\}\^\{R\}_\{ij\}\\triangleq",), ()),
    ("rai", "eq", "rotation_residual_restated"): ((r"\\mathbf\{r\}\^\{R\}_\{ij\}=",), ()),
    ("rai", "eq", "velocity_residual"): (
        (r"\\mathbf\{r\}\^\{v\}_\{ij\}", r"\\mathbf\{A\}\^\{vg\}_\{ij\}"), ()),
    ("rai", "eq", "position_residual"): (
        (r"\\mathbf\{r\}\^\{p\}_\{ij\}", r"\\mathbf\{A\}\^\{pg\}_\{ij\}"), ()),
    ("rai", "eq", "lifted_rotation_residual"): ((r"\\breve\{\\mathbf\{R\}\}_\{ij\}",), ()),
    ("rai", "eq", "lifting_definitions_Y"): ((r"\\mathbf\{Y\}\^\{\\prime\}_\{ij\}\\triangleq",), ()),
    ("rai", "eq", "isotropic_weighting"): ((r"\\tau_\{R,ij\}\^\{-1\}",), ()),
    ("rai", "eq", "u_definitions"): (
        (r"\\mathbf\{u\}\^\{v\}_\{ij\}", r"\\mathbf\{A\}\^\{vg\}_\{ij\}\\mathbf\{b\}_\{g\}"), ()),
    ("rai", "eq", "lifted_vector"): (
        (r"\\operatorname\{vec\}\(\\mathbf\{R\}_\{i\}\)", r"\\mathbf\{b\}_\{m\}"), ()),
    ("rai", "eq", "lifting_constraints"): ((r"\^\{\\top\}\\mathbf\{Y\}_\{ij\}", r"\\mathbf\{u\}\^"), ()),
    ("rai", "eq", "homogeniser"): ((r"s\^\{2\}=1",), ()),
    ("rai", "eq", "state_set"): ((r"\\mathcal\{X\}\\triangleq", r"\\mathbf\{b\}_\{m\}"), ()),
    ("rai", "redundant_constraints", "A1", "eq"): (
        (r"\\mathbf\{R\}_\{k\}\\mathbf\{R\}_\{k\}\^\{\\top\}",), ()),
    ("rai", "redundant_constraints", "B1", "eq"): ((r"\\mathbf\{w\}_\{ij\}",), ()),
    ("rai", "redundant_constraints", "B2", "eq"): (
        (r"\\mathbf\{w\}_\{ij\}", r"\\mathbf\{I\}_\{3\}"), ()),
    ("rai", "redundant_constraints", "C1", "eq"): ((r"\\mathbf\{W\}\^",), ()),

    ("delama", "eq", "semidirect_product"): ((r"XY=", r"\\mathrm\{Ad\}_\{A\}"), ()),
    ("delama", "eq", "semidirect_inverse"): (
        (r"\{X\}\^\{-1\}=", r"-\\mathrm\{Ad\}_\{\{A\}\^\{-1\}\}"), ()),
    ("delama", "eq", "element"): ((r"\\mathbb\{R\}\^\{5\\times 5\}",), ()),
    ("delama", "eq", "tangent"): ((r"\\bm\{x\}\}\^\{\\wedge\}", r"\\alpha"), ()),
    ("delama", "eq", "inverse"): ((r"\{\\mathbf\{X\}\}\^\{-1\}=", r"\\mathbf\{A\}\^\{\\top\}"), ()),
    ("delama", "eq", "adjoint"): ((r"\\mathbf\{Ad\}\^\{\\vee\}_\{\\mathbf\{X\}\}",), ()),
    ("delama", "eq", "exp"): ((r"\\exp\(\\prescript\{\}\{\}\{\\bm\{x\}\}\^\{\\wedge\}\)",), ()),
    ("delama", "eq", "log"): ((r"\\log",), ()),
    ("delama", "eq", "left_jacobian"): (
        (r"\\mathbf\{J\}_\{\\mathrm\{L\}\}\(", r"\\mathbf\{Q\}_\{1\}"), ()),
    ("delama", "eq", "endpoint"): ((r"\\mathbf\{T\}\}_\{j\}=\\mathbf\{\\Gamma\}_\{\\!ij\}",), ()),
    ("delama", "eq", "gravity_left_factor"): ((r"\\mathbf\{\\Gamma\}_\{\\!ij\}\\coloneqq",), ()),
    ("delama", "eq", "preintegrated_increment"): ((r"\\Delta\\mathbf\{R\}", r"\\Delta t_\{ij\}"), ()),
    ("delama", "eq", "mean_propagation"): (
        (r"\\mathbf\{\\Upsilon\}_\{\\!k\+1\}&=\\mathbf\{\\Upsilon\}_\{\\!k\}\\exp",), ()),
    ("delama", "eq", "state_action"): ((r"\\phi\(X_\{k\},\\xi_\{k\}\)\\coloneqq",), ()),
    ("delama", "eq", "input_action"): ((r"\\psi\(X_\{k\},u_\{k\}\)\\coloneqq",), ()),
    ("delama", "eq", "equivariant_error"): ((r"e_\{k\}\\coloneqq\\phi\(\\hat\{X\}_\{k\}\^\{-1\}",), ()),
    ("delama", "eq", "local_chart"): ((r"\\vartheta\(e_\{k\}\)\\coloneqq\\log_\{G\}",), ()),
    ("delama", "eq", "log_G"): ((r"\\log_\{G\}\\!\\left\(X_\{k\}\\right\)",), ()),
    ("delama", "eq", "error_local_coords"): (
        (r"\\mathbf\{\\Upsilon\}_\{\\!k\}\\hat\{\\mathbf\{\\Upsilon\}\}", r"\\mathbf\{J\}_\{\\"), ()),
    ("delama", "eq", "error_dynamics"): ((r"\\varepsilon\}_\{k\+1\}\\approx\\hat\{\\mathbf\{A\}\}",), ()),
    ("delama", "eq", "error_dynamics_A"): ((r"\\hat\{\\mathbf\{A\}\}_\{k\+1\}&=",), ()),
    ("delama", "eq", "error_dynamics_B"): ((r"\\hat\{\\mathbf\{B\}\}_\{k\+1\}&=",), ()),
    ("delama", "eq", "bias_correction"): (
        (r"\\hat\{\\mathbf\{\\Upsilon\}\}_\{\\!k\+1\}\^\{\+\}\\approx\\exp",), ()),
    ("delama", "exp_log_dependencies", "log_G", "eq"): ((r"\\log_\{G\}",), ()),
    ("delama", "exp_log_dependencies", "log_G", "requires_eq", "log"): ((r"\\log",), ()),
    ("delama", "exp_log_dependencies", "log_G", "requires_eq", "J_L^{-1}"): (
        (r"\\mathbf\{J\}_\{\\mathrm\{L\}\}\(",), ()),
    ("delama", "exp_log_dependencies", "error_local_coords", "eq"): (
        (r"\\mathbf\{\\Upsilon\}_\{\\!k\}\\hat\{\\mathbf\{\\Upsilon\}\}",), ()),
    ("delama", "exp_log_dependencies", "error_local_coords", "requires_eq", "log"): ((r"\\log",), ()),
    ("delama", "exp_log_dependencies", "error_local_coords", "requires_eq", "J_L^{-1}"): (
        (r"\\mathbf\{J\}_\{\\mathrm\{L\}\}\(",), ()),

    ("barfoot", "eq", "so3"): ((r"SO\(3\)=", r"\\mbox\{det\}"), ()),
    ("barfoot", "eq", "se3_element"): ((r"SE\(3\)=", r"\\mathbf\{C\}&\\mathbf\{r\}"), ()),
    ("barfoot", "eq", "se3_tangent"): ((r"\{\\boldsymbol\{\\rho\}\}\\\\ \{\\boldsymbol\{\\phi\}\}",), ()),
    ("barfoot", "eq", "adjoint"): ((r"\\mbox\{Ad\}\\left\(\\mathbf\{T\}\\right\)",), ()),
    ("barfoot", "eq", "cayley"): ((r"\\mbox\{cay\}\(\{\\boldsymbol\{\\phi\}\}\^\{\\wedge\}\)",), ()),
    ("barfoot", "eq", "cayley_inverse"): ((r"\\mbox\{cay\}\^\{-1\}",), ()),
    ("barfoot", "eq", "cayley_se3"): ((r"\\mbox\{cay\}\(\{\\boldsymbol\{\\xi\}\}\^\{\\wedge\}\)",), ()),
    ("barfoot", "eq", "nonconvex"): ((r"f\(\\mathbf\{z\}\)", r"g_\{i\}\(\\mathbf\{z\}\)=0"), ()),
    ("barfoot", "eq", "homogeneous_qcqp"): ((r"\\mathbf\{A\}_\{0\}\\mathbf\{x\}=1",), ()),
    ("barfoot", "eq", "rank1_sdp"): ((r"\\mbox\{rank\}\(\\mathbf\{X\}\)=1",), ()),
    ("barfoot", "eq", "shor_relaxation"): ((r"\\mathbf\{X\}\\succeq 0",), (r"\\mbox\{rank\}",)),

    ("gtsam", "eq", "state"): (
        (r"\\mathcal M=\\mathrm\{Gal\}\(3\)\\times\\mathbb R\^6", r"b_\{a,k\}"), ()),
    ("gtsam", "eq", "element"): ((r"X=\(R,v,p,t\)", r"R & v & p"), ()),
    ("gtsam", "eq", "product"): ((r"p_1\+R_1p_2\+t_2v_1",), ()),
    ("gtsam", "eq", "inverse"): ((r"-R\^\\top\(p-tv\)",), ()),
    ("gtsam", "eq", "tangent"): ((r"x=\(\\omega,\\nu,\\rho,\\alpha\)",), ()),
    ("gtsam", "eq", "exp"): ((r"\\operatorname\{Exp\}\(x\)", r"\\alpha\\Gamma_2\(\\omega\)\\nu"), ()),
    ("gtsam", "eq", "adjoint"): ((r"\\operatorname\{Ad\}_X", r"\(p-tv\)\^\\wedge R"), ()),
    ("gtsam", "eq", "imu_sample"): ((r"u_k=", r"a_k", r"\\omega_k"), ()),
    ("gtsam", "eq", "composition"): ((r"F_h\(\(\\Upsilon_k,\\beta_k\),u_k\)",), ()),
    ("gtsam", "eq", "held_input_composition"): (
        (r"\\hat\\Upsilon_\{i,k\+1\}=\\hat\\Upsilon_\{ik\}V_k",), ()),
    ("gtsam", "eq", "gravity_left_factor"): ((r"W_\{ij\}=\\operatorname\{NavState\}",), ()),
    ("gtsam", "eq", "preintegrated_increment"): (
        (r"U_\{ij\}\(\\beta_i\)=\\operatorname\{NavState\}",), ()),
    ("gtsam", "eq", "endpoint"): ((r"X_j=W_\{ij\}", r"\\phi_\{\\Delta t_\{ij\}\}\(X_i\)"), ()),
    ("gtsam", "eq", "bias_lift_E"): ((r"E=", r"\\mathbb R\^\{10\\times6\}"), ()),
    ("gtsam", "eq", "error"): ((r"\\hat\\xi_k\^\{-1\}\\xi_k",), ()),
    ("gtsam", "eq", "error_vector"): ((r"\\epsilon_k=\(e_k,\\delta\\beta_k\)",), ()),
    ("gtsam", "eq", "C_matrix"): ((r"C=J_R\(x\)hE",), ()),
    ("gtsam", "eq", "transition"): ((r"A_\{\\mathrm\{LI\}\}", r"\\operatorname\{Ad\}_\{V\^\{-1\}\}"), ()),
    ("gtsam", "eq", "bias_correction"): ((r"\\Upsilon_k\(\\hat\\beta\+\\delta\\beta\)",), ()),
    ("gtsam", "eq", "bias_jacobian_recursion"): (
        (r"J_\{k\+1\}=\\operatorname\{Ad\}_\{V_k\^\{-1\}\}J_k-C_k",), ()),
    ("gtsam", "eq", "bias_correction_restated"): ((r"\\Upsilon_k\(\\beta\)\\simeq",), ()),
    ("gtsam", "eq", "navstate_tangent"): ((r"n=\(\\delta\\theta,\\delta p,\\delta v\)",), ()),
    ("gtsam", "eq", "navstate_projection"): ((r"P_N=", r"\\mathbb R\^\{9\\times10\}"), ()),
    ("gtsam", "eq", "residual"): ((r"r_\{ij\}=\\operatorname\{Local\}", r"logmap"), ()),
    ("gtsam", "eq", "rotating_frame"): ((r"W\^\\Omega_\{ij\}", r"\\mathcal L_\\Omega\(X_i\)"), ()),
}

#: Entries whose fact the source states in prose or an unnumbered display.  The regex must
#: find the matrix document's own record of why no equation number exists.
SENTINEL_JUSTIFICATION: dict[tuple[str, ...], str] = {
    ("rai", "eq", "bias_ordering"): r"bias ordering `b = \[b_g\^T b_a\^T\]\^T`",
    ("rai", "eq", "rank1_criterion"): r"log-SVR .{0,8} 5 rank-1 threshold",
    ("delama", "eq", "bias_process_noise"): r"equation number for `Q_d`",
    ("delama", "eq", "continuous_time_ins"): r"continuous-time biased INS",
    ("barfoot", "eq", "rank1_criterion"): r"log-SVR .{0,8} 5 rank-1 threshold",
    ("barfoot", "eq", "sparsity"): r"measurement-sparsity/tightness",
    ("barfoot", "eq", "licq_failure"): r"LICQ failure under redundant constraints",
}


def _walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, path + (key,))
    else:
        yield path, node


def _is_eq_path(path) -> bool:
    return path[0] == "eq" or path[-1] == "eq"


def _check_source_row(key: str) -> None:
    row = CONVENTIONS[key]
    checked = set()
    for path, (require, forbid) in EQ_SIGNATURES.items():
        if path[0] != key:
            continue
        value = table_value(path)
        assert value != COULD_NOT_EXTRACT, (
            f"CONVENTIONS{list(path)} is recorded as {COULD_NOT_EXTRACT!r}, but "
            f"{SOURCE_FIXTURES[key][1]} does number this equation"
        )
        assert_equation_says(key, value, require, forbid)
        checked.add(path[1:])
    for path, pattern in SENTINEL_JUSTIFICATION.items():
        if path[0] != key:
            continue
        assert table_value(path) == COULD_NOT_EXTRACT, (
            f"CONVENTIONS{list(path)} == {table_value(path)!r}; the matrix document records "
            f"this item as unnumbered, so it must be {COULD_NOT_EXTRACT!r}"
        )
        assert re.search(pattern, matrix_doc(), re.S), (
            f"docs/00_literature_matrix.md no longer records why {list(path)} is unnumbered"
        )
        checked.add(path[1:])

    # No eq entry may escape: each is either signed against the source or justified as prose.
    for path, value in _walk(row):
        if _is_eq_path(path):
            assert path in checked, f"CONVENTIONS[{key!r}]{list(path)} is not checked at all"

    # Free prose must be traceable to the deriver's matrix, and any equation it cites must be
    # a real equation of the source it cites.
    for path, value in _walk(row):
        where = f"CONVENTIONS[{key!r}]" + "".join(f"[{p!r}]" for p in path)
        if _is_eq_path(path):
            continue
        items = ()
        if isinstance(value, str):
            items = (value,)
        elif isinstance(value, tuple) and value and all(isinstance(x, str) for x in value):
            items = value
        for item in items:
            assert_equation_mentions_resolve(where, item, (key,))
            if len(item) > 24 and item != COULD_NOT_EXTRACT:
                assert_corroborated(where, item, floor=0.5)

    assert row["source_url"] == SOURCE_FIXTURES[key][1], (
        f"CONVENTIONS[{key!r}]['source_url'] is not the URL the archived fixture came from"
    )
    tokens = re.findall(r"arXiv:[\d.]+v\d+|10\.\d{4,}/[\w.]+|\d+\.\d+\.\d+", row["citation"])
    assert tokens, f"CONVENTIONS[{key!r}]['citation'] carries no identifier at all"
    for token in tokens:
        assert token in matrix_doc(), (
            f"CONVENTIONS[{key!r}]['citation'] cites {token!r}, which the literature matrix "
            f"does not record"
        )


@pytest.mark.claim("M0-C02")
@pytest.mark.claim("M0-C04")
@pytest.mark.claim("M0-C05")
@pytest.mark.claim("M0-C06")
@pytest.mark.claim("M0-C10")
@pytest.mark.claim("M0-C12")
def test_rai_row_is_sourced():
    _check_source_row("rai")


@pytest.mark.claim("M0-C07")
@pytest.mark.claim("M0-C08")
@pytest.mark.claim("M0-C09")
@pytest.mark.claim("M0-C10")
@pytest.mark.claim("M0-C11")
@pytest.mark.claim("M0-C15")
def test_delama_row_is_sourced():
    _check_source_row("delama")


@pytest.mark.claim("M0-C10")
@pytest.mark.claim("M0-C12")
@pytest.mark.claim("M0-C13")
def test_barfoot_row_is_sourced():
    _check_source_row("barfoot")


@pytest.mark.claim("M0-C10")
@pytest.mark.claim("M0-C11")
@pytest.mark.claim("M0-C13")
@pytest.mark.claim("M0-C14")
def test_gtsam_row_is_sourced():
    _check_source_row("gtsam")


@pytest.mark.claim("M0-C13")
def test_incompatibility_register_is_sourced():
    for key, row in INCOMPATIBILITIES.items():
        for path, value in _walk(row):
            where = f"INCOMPATIBILITIES[{key!r}]" + "".join(f"[{p!r}]" for p in path)
            side = path[0] if path[0] in ("a", "b") else None
            cites = (row[side]["source"],) if side else (row["a"]["source"], row["b"]["source"])
            if path[-1] == "eq":
                if value == COULD_NOT_EXTRACT:
                    continue
                index = source_equations(row[path[0]]["source"]) if side else source_equations(
                    row["a"]["source"]
                )
                assert value in index or re.sub(r"[a-z]+$", "", value) in index, (
                    f"{where} == {value!r} is not an equation of the source it cites"
                )
            elif isinstance(value, str):
                assert_equation_mentions_resolve(where, value, cites)
                if len(value) > 24:
                    assert_corroborated(where, value, floor=0.5)


# ======================================================================================
# Structural invariants of the table itself (contract section 8, rules 1-3).
# Not bound to a numbered claim: it guards the shape every other test reads.
# ======================================================================================

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
