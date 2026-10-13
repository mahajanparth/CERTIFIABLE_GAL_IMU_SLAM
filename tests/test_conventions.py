"""M0 -- one test per claim in ``docs/claims/M0.yaml``.

Every claim M0-C01..M0-C15 is bound with ``@pytest.mark.claim(...)``; that marker is the
only channel the orchestrator reads, and an unbound claim reads UNTESTED and blocks the gate.

Why this file looks the way it does (``docs/verification/m0-literature-matrix.md``, F-1..F-7)
------------------------------------------------------------------------------------------
The first version restated the contents of :mod:`src.conventions` as literals in the test
body and compared the table against those literals; 40 of 41 deliberately corrupted table
fields still passed.  The rule now observed throughout:

    **No expectation about the table is written in this file.**  Every assertion reads a
    value out of ``CONVENTIONS`` / ``INCOMPATIBILITIES`` and decides it against something a
    mutation of the table cannot move:

    1. the **claim file** ``docs/claims/M0.yaml`` -- the statement (and, where the statement
       defers to it, the ``domain`` / ``test_hint``) is parsed, and every
       ``<table path> == <literal>`` predicate in it is checked (:func:`assert_claim_statement`);
    2. the **archived primary sources** under ``tests/fixtures`` -- each equation label the
       table records is looked up in the source's *own* numbering and the equation found
       there must say what the label is recorded for (:data:`EQ_SIGNATURES`); orderings,
       coefficients, constraint sets and verbatim quotes are read straight out of the
       authors' LaTeX (:func:`canon`, :func:`bmatrices`);
    3. the **installed gtsam 4.3.0**, exercised at runtime;
    4. **execution of the recorded string itself** -- recorded formulas are parsed and run,
       so a corrupted sign or slot produces a wrong *number*;
    5. **cached excerpts of documents the HTML renders do not reproduce** -- a committed text
       excerpt of the Barfoot PDF (for the sub-labels the HTML drops) and the Crossref record
       of its journal version, each with its provenance in ``tests/fixtures/provenance.json``.

The equation index never invents a label: it holds exactly the tags each archived render
prints (M0-F07).  ``tests/test_mutation_m0.py`` (claims M0-F01/F06) corrupts the 41 fields of
pass 1, the 18 survivors of pass 2 and the fields added since, and requires this file to fail
for every one of them.  Fields no source can decide are listed, not silently skipped, in
``src.conventions.UNVERIFIED_COMMENTARY``.

Assertions that could not fail (F-7, M0-F08) were deleted, not reworded; each deletion is
marked in place with the reason.
"""
from __future__ import annotations

import ast
import functools
import gzip
import html as _html
import json
import math
import operator
import pathlib
import re
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
# an lvalue: optional CONVENTIONS["src"], optional bare field name, then ["k"]["k"]...
_LVALUE_CHAIN = re.compile(
    r'(?:CONVENTIONS\s*\[\s*"(\w+)"\s*\]\s*)?(\b[A-Za-z_]\w*)?\s*((?:\[\s*"[^"\]]+"\s*\]\s*)+)$'
)
_LVALUE_NAME = re.compile(r"(\b[A-Za-z_]\w*)\s*$")
_RVALUE = re.compile(
    r"\s*(?:"
    r'"(?P<str>[^"]*)"'
    r"|(?P<tuple>\((?:[^()]|\([^()]*\))*\))"
    r"|(?P<const>True|False|None)\b"
    r"|(?P<num>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r")"
)

#: Bare names a claim statement uses for a nested field.  Only the *path* is resolved here;
#: the expected value still comes from the statement text.
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


def statement_expectations(cid: str, field: str = "statement"):
    """Every ``<table path> == <literal>`` / ``<path> is <const>`` predicate in a claim field."""
    text = claim(cid)[field]
    found: list[tuple[tuple[str, ...], object]] = []
    for m in re.finditer(r"(==|\bis\b)", text):
        head, tail = text[: m.start()], text[m.end():]
        rv = _RVALUE.match(tail)
        if rv is None:
            continue
        if rv.group("str") is not None:
            expected: object = rv.group("str")
        elif rv.group("tuple") is not None:
            expected = eval(rv.group("tuple"), {"__builtins__": {}}, {"slice": slice})  # noqa: S307
        elif rv.group("const") is not None:
            expected = {"True": True, "False": False, "None": None}[rv.group("const")]
        else:
            expected = ast.literal_eval(rv.group("num"))

        prefixes = list(_CONV_PREFIX.finditer(head))
        src_ctx = prefixes[-1].group(1) if prefixes else None

        chain = _LVALUE_CHAIN.search(head.rstrip())
        if chain:
            keys = re.findall(r'\[\s*"([^"\]]+)"\s*\]', chain.group(3))
            src = chain.group(1) or src_ctx
            if chain.group(1) is None and chain.group(2):
                keys = [chain.group(2), *keys]
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


def assert_claim_statement(cid: str, field: str = "statement", minimum: int = 1) -> int:
    """Check every table predicate the claim spells out.  Expected values come from the YAML."""
    exp = statement_expectations(cid, field)
    assert len(exp) >= minimum, (
        f"{cid}: only {len(exp)} table predicates parsed out of the claim's {field!r}"
    )
    for path, expected in exp:
        got = table_value(path)
        assert got == expected, (
            f"{cid}: CONVENTIONS{''.join(f'[{k!r}]' for k in path)} == {got!r}, but the "
            f"claim's {field} in docs/claims/M0.yaml says {expected!r}"
        )
    return len(exp)


def claim_number(cid: str, field: str, pattern: str) -> float:
    """Pull a numeric bound out of the claim's own text."""
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
def source_raw(key: str) -> str:
    """The archived source document, byte-for-byte as fetched.

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
            req = urllib.request.Request(url, headers={"User-Agent": "cgis-m0/0.1"})
            with urllib.request.urlopen(req, timeout=60) as fh:
                raw = fh.read()
        except Exception as exc:
            raise AssertionError(
                f"{key} source text unavailable: no cached fixture at {path} and the fetch "
                f"of {url} failed ({exc!r}). The claim must FAIL, not skip."
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(raw, 9))
    return raw.decode("utf-8", "replace")


@functools.lru_cache(maxsize=8)
def source_html(key: str) -> str:
    """The source with ``<script>`` / ``<style>`` stripped (for raw-substring scans)."""
    text = re.sub(r"(?is)<script.*?</script>", " ", source_raw(key))
    return re.sub(r"(?is)<style.*?</style>", " ", text)


@functools.lru_cache(maxsize=8)
def source_prose(key: str) -> str:
    """Running text of an arXiv source, each formula replaced by ``$<its LaTeX>$``."""
    text = re.sub(
        r"(?s)<math\b.*?</math>",
        lambda m: " $" + _html.unescape(
            (re.search(r'alttext="([^"]*)"', m.group(0)) or [None, ""])[1]) + "$ ",
        source_html(key),
    )
    text = _html.unescape(re.sub(r"<[^>]+>", " ", text))
    return " ".join(text.split())


_LATEXML_BLOCK = re.compile(r'<table\b[^>]*\bid="[SA][^"]*?\.E(Gx)?([0-9]*)"[^>]*>')
_LATEXML_ROW = re.compile(r'<tr\b([^>]*\bclass="[^"]*ltx_eqn_row[^"]*"[^>]*)>')
_LATEXML_TAG = re.compile(r"ltx_tag_equation(?:group)?[^>]*>\(([^<)]*)\)<")
_LATEXML_ALT = re.compile(r'alttext="([^"]*)"')
_LATEXML_TD = re.compile(r"<td\b([^>]*)>")


def _table_end(raw: str, start: int) -> int:
    depth, i = 1, start
    while depth:
        nxt, close = raw.find("<table", i), raw.find("</table>", i)
        if nxt != -1 and nxt < close:
            depth, i = depth + 1, nxt + 6
        else:
            depth, i = depth - 1, close + 8
    return i


def _latexml_row(attrs: str, body: str) -> dict | None:
    r"""One ``<tr class="ltx_eqn_row">``: its printed tags, its LaTeX, whether it has a
    left-hand side (an aligned row whose first math cell is the ``align_left`` one is a
    continuation line such as ``\qquad - ...``), and LaTeXML's row id."""
    cells = []
    for td in _LATEXML_TD.finditer(body):
        nxt = body.find("<td", td.end())
        seg = body[td.end(): nxt if nxt != -1 else len(body)]
        cells.append((td.group(1), " ".join(_html.unescape(a) for a in _LATEXML_ALT.findall(seg))))
    if any("white-space:normal" in a for a, _ in cells):
        return None                     # an \intertext line: prose, not an equation row
    tags = _LATEXML_TAG.findall(body)
    latex = " ".join(lx for _, lx in cells if lx.strip())
    if not tags and not latex.strip():
        return None
    math = [a for a, lx in cells if lx.strip()]
    rid = re.search(r'\bid="([^"]*)"', attrs)
    return {"tags": tags, "latex": latex, "id": rid.group(1) if rid else "",
            "continuation": bool(math) and "ltx_align_left" in math[0]}


def _latexml_equations(raw: str) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]]]:
    r"""Row-resolved LaTeXML index that never invents a label (M0-F07).

    Returns ``(index, blocks, unlabelled)``:

    * ``index`` -- ``{tag -> LaTeX}`` for exactly the tags the HTML prints in
      ``ltx_tag_equation`` cells.  A label is assigned content only by what LaTeXML itself
      records: the tagged row; the rows LaTeXML ids ``<eq>Xa``, ``<eq>Xb``... (continuation
      rows of the single-numbered display ``<eq>X``); untagged continuation lines that follow
      a tagged row; and an untagged head row immediately followed by a tagged continuation
      line (the ``r = ... \nonumber \\ \qquad - ... (25a)`` idiom).
    * ``blocks`` -- ``{block number -> LaTeX of every row}``, so a bare block label such as
      Barfoot "(15)" (how the authors cite it) denotes the whole display.
    * ``unlabelled`` -- ``{block number -> [LaTeX of rows the HTML prints without a tag]}``.
      These rows get NO label here, ever (the old index numbered them by row position and
      thereby invented barfoot (14a)/(15a)/(16a)/(30b) and rai (25c)).
    """
    index: dict[str, str] = {}
    blocks: dict[str, str] = {}
    unlabelled: dict[str, list[str]] = {}
    for m in _LATEXML_BLOCK.finditer(raw):
        base = "" if m.group(1) else m.group(2)
        body = raw[m.end():_table_end(raw, m.end())]
        heads = list(_LATEXML_ROW.finditer(body))
        rows = []
        for k, h in enumerate(heads):
            stop = heads[k + 1].start() if k + 1 < len(heads) else len(body)
            row = _latexml_row(h.group(1), body[h.end():stop])
            if row:
                rows.append(row)
        out: dict[str, list[str]] = {}
        orphan: list[str] = []
        pending: list[str] = []
        last: list[str] = []
        for row in rows:
            if row["tags"]:
                content = [*pending, row["latex"]] if (row["continuation"] and pending) else [row["latex"]]
                if not (row["continuation"] and pending):
                    orphan.extend(pending)
                pending = []
                for t in row["tags"]:
                    out[t] = list(content)
                last = row["tags"]
            elif last and not pending and (row["continuation"] or re.search(r"X[a-z]+$", row["id"])):
                for t in last:
                    out[t].append(row["latex"])
            else:
                pending.append(row["latex"])
        orphan.extend(pending)
        for t, parts in out.items():
            index.setdefault(t, " ".join(parts))
        if base:
            blocks.setdefault(base, " ".join(r["latex"] for r in rows))
            if orphan:
                unlabelled.setdefault(base, orphan)
    return index, blocks, unlabelled


def _myst_equations(raw: str) -> dict[str, str]:
    """The GTSAM doc is a mystmd page; its AST (with each equation's ``enumerator`` and raw
    LaTeX ``value``) is the JSON in ``window.__remixContext`` -- inside a ``<script>``, which
    is why the stripped text indexed zero equations."""
    m = re.search(r"window\.__remixContext\s*=\s*", raw)
    assert m, "the archived GTSAM page no longer embeds its mystmd AST"
    doc, _ = json.JSONDecoder().raw_decode(raw[m.end():])
    index: dict[str, str] = {}

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "math" and "enumerator" in node:
                index.setdefault(str(node["enumerator"]), node["value"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(doc)
    return index


@functools.lru_cache(maxsize=8)
def _source_index(key: str):
    raw = source_raw(key)
    if key == "gtsam":
        return _myst_equations(raw), {}, {}
    return _latexml_equations(raw)


@functools.lru_cache(maxsize=8)
def source_equations(key: str) -> dict[str, str]:
    """``{equation label -> the authors' own LaTeX}`` -- only labels the source prints."""
    return {k: " ".join(v.split()) for k, v in _source_index(key)[0].items()}


# -- labels the HTML render drops but a cached text excerpt of the PDF prints ------------

#: ``source key -> committed text excerpt of that source's PDF`` (never the PDF itself).
PDF_EXCERPTS = {"barfoot": "barfoot_2308.12418v2_pdf_p4_cayley.txt"}
_PDF_TAG = re.compile(r"\((\d+[a-z])\)")
_UNICODE_SYMBOLS = {"ϕ": "phi", "ξ": "xi", "φ": "varphi"}


@functools.lru_cache(maxsize=4)
def pdf_excerpt(key: str) -> tuple[dict[str, str], str]:
    """``({printed label -> canonical LHS printed in that label's display}, provenance)``.

    The window of a label is the excerpt text after the previous printed label and up to
    this one; its LHS is the line ``<lhs>   =`` in that window (pdftotext -layout keeps the
    display's alignment columns apart)."""
    text = (_FIXTURES / PDF_EXCERPTS[key]).read_text(encoding="utf-8")
    header = "\n".join(line for line in text.splitlines() if line.startswith("#"))
    body = text.split("# ---- excerpt begins ----", 1)[1].split("# ---- excerpt ends ----", 1)[0]
    found: dict[str, str] = {}
    prev = 0
    for m in _PDF_TAG.finditer(body):
        window = body[prev:m.start()]
        prev = m.end()
        lhs = re.findall(r"^\s*(\S+(?:\s?\(\S\))?)\s+=", window, re.M)
        if not lhs:
            continue
        sym = lhs[-1].replace(" ", "")
        for u, name in _UNICODE_SYMBOLS.items():
            sym = sym.replace(u, name)
        found[m.group(1)] = sym
    return found, header


def _pdf_label_latex(key: str, label: str) -> str | None:
    """The HTML row a PDF-printed label denotes: the UNIQUE row of block ``<n>`` that the
    HTML prints without a tag and whose LHS is the LHS the PDF prints beside ``(<n><x>)``."""
    if key not in PDF_EXCERPTS:
        return None
    printed, _ = pdf_excerpt(key)
    if label not in printed:
        return None
    base = re.match(r"\d+", label).group(0)
    rows = _source_index(key)[2].get(base, [])
    hits = [r for r in rows if canon(r).split("=", 1)[0] == printed[label]]
    assert len(hits) == 1, (
        f"{key} ({label}) is printed in the PDF excerpt beside LHS {printed[label]!r}, but block "
        f"({base}) of the HTML has {len(hits)} untagged rows with that LHS"
    )
    return " ".join(hits[0].split())


def equation_latex(key: str, label: str) -> str:
    """The LaTeX printed at ``label``.  A bare block label (``"26"``) for a sub-numbered block
    (``26a``, ``26b``) denotes the whole display.  A sub-label the HTML render drops is
    resolved only through a committed PDF text excerpt that prints it (:data:`PDF_EXCERPTS`)."""
    index = source_equations(key)
    if label in index:
        return index[label]
    via_pdf = _pdf_label_latex(key, label)
    if via_pdf is not None:
        return via_pdf
    blocks = _source_index(key)[1]
    subs = [k for k in index if re.fullmatch(re.escape(label) + r"[a-z]", k)]
    assert subs and label in blocks, (
        f"{key} has no equation labelled ({label}) -- the archived source "
        f"{SOURCE_FIXTURES[key][1]} numbers {len(index)} equations and this is not one"
    )
    return " ".join(blocks[label].split())


def assert_equation_says(key: str, label: str, require, forbid=()) -> None:
    latex = equation_latex(key, label)
    for pattern in require:
        assert re.search(pattern, latex), (
            f"{key} Eq. ({label}) does not contain {pattern!r}; the source prints\n  {latex[:400]}"
        )
    for pattern in forbid:
        assert not re.search(pattern, latex), f"{key} Eq. ({label}) contains {pattern!r}"


# -- reading LaTeX as plain symbols -----------------------------------------------------

_LX_DROP = re.compile(
    r"\\displaystyle|\\!|\\,|\\;|\\qquad|\\quad|\\left(?![a-z])|\\right(?![a-z])|\\bigl|"
    r"\\bigr|\\Biggl|\\Biggr|\\Big|\\vskip-?[\d.]+pt|\\prescript\{\}\{\}|\\scriptscriptstyle|"
    r"\\boxed"
)
_LX_WRAP = re.compile(
    r"\\(?:mathbf|bm|boldsymbol|mathrm|mathcal|mathbb|operatorname|text|texttt|mbox|mathsf|"
    r"mathring)\{([^{}]*)\}"
)
_LX_ACCENT = re.compile(r"\\(bar|hat|tilde|breve)\{([^{}]*)\}")


def canon(latex: str) -> str:
    """LaTeX reduced to its symbols: fonts and spacing dropped, ``\\bar{R}`` -> ``Rbar``,
    ``\\tfrac{1}{2}`` -> ``(1/2)``, ``^{\\wedge}`` -> ``^``, ``\\top`` -> ``T``, greek letters
    spelled out, braces and whitespace removed."""
    s = _LX_DROP.sub("", latex)
    s = re.sub(r"\\(?:mathcal|mathbb|mathbf|mathsf|mathrm)\s+(?=\w)", "", s)   # \mathbb R
    s = re.sub(r"\\coloneqq|\\triangleq", "=", s)
    prev = None
    while prev != s:
        prev, s = s, _LX_ACCENT.sub(lambda m: m.group(2) + m.group(1), _LX_WRAP.sub(r"\1", s))
    s = re.sub(r"\\t?frac\{1\}\{2\}|\\tfrac12", "(1/2)", s)
    s = re.sub(r"(?:\{\})?\^\{?\\wedge\}?", "^", s)
    s = re.sub(r"\^\{?\\vee\}?", "^v", s)
    s = (s.replace("\\|", "||").replace("\\top", "T").replace("\\prime", "'")
         .replace("\\times", "x"))
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    s = re.sub(r"\\(?=[^A-Za-z]|$)", "", s)
    s = re.sub(r"\s+", "", s.replace("{", "").replace("}", ""))
    return s.replace("||_2", "||")


def canon_table(text: str) -> str:
    """A table formula string reduced to the same alphabet as :func:`canon`."""
    s = text.replace("**", "^").replace("==", "=").replace("*", "")
    s = re.sub(r"\s+x\s+", "x", s)
    return re.sub(r"\s+", "", s.replace("{", "").replace("}", ""))


def bmatrices(latex: str) -> list[list[list[str]]]:
    """Every ``bmatrix`` in an equation, as rows of canonical cells."""
    out = []
    for m in re.finditer(r"\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}", latex, re.S):
        rows = [r for r in re.split(r"\\\\", m.group(1)) if r.strip()]
        out.append([[canon(c) for c in r.split("&")] for r in rows])
    return out


def element_order_from(latex: str) -> tuple[str, ...]:
    """Slot names of a homogeneous group element read off its matrix: the first block row,
    then the time entry if the second row carries one."""
    M = bmatrices(latex)[-1]
    order = list(M[0])
    if len(M) > 1 and not re.fullmatch(r"[\d.]+", M[1][-1]):
        order.append(M[1][-1])
    return tuple(order)


def tuple_after(text: str, lhs: str) -> tuple[str, ...]:
    """``lhs=(a,b,c)`` in canonical text -> ``("a","b","c")``."""
    m = re.search(re.escape(lhs) + r"=\(([^()]*)\)", text)
    assert m, f"no tuple {lhs}=(...) in {text[:200]!r}"
    return tuple(m.group(1).split(","))


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


def mathify(text: str, scalars=(), functions=()) -> str:
    """Turn a recorded formula such as ``-A^T (b - c a)`` into Python.

    Juxtaposition becomes ``@`` between matrix/vector operands and ``*`` when either side is
    a scalar; ``X^T`` is a transpose, ``X^-1`` an inverse, a trailing ``^`` the skew operator,
    ``||x||`` a norm.
    """
    s = re.sub(r"\((\d+)/(\d+)\)", lambda m: repr(int(m.group(1)) / int(m.group(2))), text)
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
            if left and right and not (prev in functions and tok == "("):
                out.append("*" if (_is_scalar(prev, scalars) or _is_scalar(tok, scalars))
                           else "@")
        out.append(tok)
    # an explicit "*" between two non-scalar operands is a matrix product
    for i, tok in enumerate(out):
        if tok == "*" and 0 < i < len(out) - 1 and not (
                _is_scalar(out[i - 1], scalars) or _is_scalar(out[i + 1], scalars)
                or out[i - 1] == ")" or out[i + 1] == "("):
            out[i] = "@"
    s = " ".join(out)
    s = re.sub(r"TR_(\w+)", r"(\1).T", s)
    s = re.sub(r"INV_(\w+)", r"_inv(\1)", s)
    s = re.sub(r"HAT_(\w+)", r"_hat(\1)", s)
    return re.sub(r"NORM_(\w+)", r"_norm(\1)", s)


def run_formula(text: str, env: dict, scalars=(), functions=()):
    """Evaluate a formula recorded in the conventions table (or written in a claim)."""
    expr = mathify(text, scalars, functions)
    ns = dict(env)
    ns.update(_inv=_inv, _hat=_hat, _norm=_norm, np=np)
    try:
        return np.asarray(eval(expr, {"__builtins__": {}}, ns), dtype=float)  # noqa: S307
    except Exception as exc:
        raise AssertionError(
            f"the recorded formula {text!r} does not evaluate ({expr!r}): {exc!r}"
        ) from exc


def gal_from_rows(text: str, env: dict) -> np.ndarray:
    """Build a 5x5 Gal(3) matrix from a recorded ``[[I, a, b],[0,1,c],[0,0,1]]`` string."""
    rows = [r for r in re.findall(r"\[([^][]*)\]", text) if r.strip()]
    assert len(rows) == 3, f"{text!r} is not a 3-row 5x5 block matrix"
    ns = dict(env)
    ns.update(np=np, I=np.eye(3))
    # a claim writes powers as "^", the table as "**"; inside a numeric cell both are powers
    cells = [[c.strip().replace("**", "^").replace("^", "**") for c in row.split(",")]
             for row in rows]
    assert [len(c) for c in cells] == [3, 3, 3], f"{text!r} is not 3x3 blocks"
    M = np.zeros((5, 5))
    try:
        top = [np.asarray(eval(c, {"__builtins__": {}}, ns), dtype=float)    # noqa: S307
               for c in cells[0]]
        # a literal 0 stands for a zero block; any other scalar in a block slot is an error
        top = [np.zeros(shape) if (t.shape == () and t == 0.0) else t
               for t, shape in zip(top, ((3, 3), (3,), (3,)))]
        M[0:3, 0:3] = top[0].reshape(3, 3)
        M[0:3, 3] = top[1].reshape(3)
        M[0:3, 4] = top[2].reshape(3)
        for r, row in zip((3, 4), cells[1:]):
            for c, cell in zip((3, 4), row[1:]):
                M[r, c] = float(eval(cell, {"__builtins__": {}}, ns))         # noqa: S307
    except Exception as exc:
        raise AssertionError(f"{text!r} does not evaluate as a Gal(3) matrix: {exc!r}") from exc
    return M


def gal_from_latex(cells: list[list[str]], env: dict) -> np.ndarray:
    """Build a 5x5 matrix from a source ``bmatrix`` (canonical cells), e.g. Delama Eq. (18)."""
    def py(cell: str) -> str:
        s = cell.replace("Deltat_ij", " dt ").replace("I_3", " I ").replace("0_1x3", "0")
        s = s.replace("^2", "**2")
        return mathify(s, scalars=("dt", "1", "2"))
    return gal_from_rows("[" + "],[".join(", ".join(py(c) for c in row) for row in cells) + "]",
                         env)


# ======================================================================================
# Shared helpers
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


_SLOT_WIDTH = {"rotation": 3, "velocity": 3, "position": 3, "time": 1}


def _tangent_blocks(slots) -> dict[str, slice]:
    """Map recorded slot names onto their blocks of a 10-vector, in the recorded order."""
    out, at = {}, 0
    for slot in slots:
        out[slot] = slice(at, at + _SLOT_WIDTH[slot])
        at += _SLOT_WIDTH[slot]
    assert at == 10, f"recorded tangent slots {slots} do not fill R^10"
    return out


def _rand_gal(rng):
    return gal(Rotation.random(1, rng=rng).as_matrix()[0], rng.standard_normal(3),
               rng.standard_normal(3), float(rng.standard_normal()))


def _slots_from_hat(hat_rows, element_order, element_slots, tangent_order):
    """Which element slot each tangent coordinate lands in, read off a source hat matrix:
    ``x^`` puts the rotation coordinate under a skew, each vector coordinate in the column
    of the element slot it perturbs, and the time coordinate in the time entry."""
    slot_of_col = dict(zip(range(len(element_order)), element_slots))
    where = {}
    for k, cell in enumerate(hat_rows[0]):
        if cell.endswith("^"):
            where[cell[:-1]] = "rotation"
        elif k:
            where[cell] = slot_of_col[k]
    if len(hat_rows) > 1 and len(element_order) > len(hat_rows[0]):
        where[hat_rows[1][-1]] = slot_of_col[len(element_order) - 1]
    return tuple(where[name] for name in tangent_order)


# ======================================================================================
# M0-C01
# ======================================================================================

@pytest.mark.claim("M0-C01")
def test_rai_group_is_so3_only_and_source_text_is_galilean_free():
    assert_claim_statement("M0-C01", minimum=3)   # group, element_order, tangent_order
    rai = CONVENTIONS["rai"]
    html = source_html("rai")

    # v_i and p_i never leave R^n: the recorded space must be the one the paper declares.
    dims = {}
    for field in ("velocity_space", "position_space"):
        dim = re.fullmatch(r"R\^(\d+)", rai[field])
        assert dim, f"rai[{field!r}] == {rai[field]!r} is not of the form R^n"
        dims[field] = int(dim.group(1))
        assert rf"\mathbf{{v}}_{{i}},\mathbf{{p}}_{{i}}\in\mathbb{{R}}^{{{dims[field]}}}" in html, (
            f"the rai source never declares v_i, p_i in {rai[field]!r}"
        )
    # ... and the per-keyframe state is SO(n) x R^n x R^n, nothing else.
    n = re.fullmatch(r"SO\((\d+)\)", rai["group"])
    assert n, rai["group"]
    so_dim = int(n.group(1)) * (int(n.group(1)) - 1) // 2
    assert rai["state_dim"] == so_dim + dims["velocity_space"] + dims["position_space"], (
        f"rai state_dim == {rai['state_dim']} is not dim {rai['group']} + dim v + dim p"
    )
    assert_equation_says("rai", rai["eq"]["state_set"],
                         require=[r"\\mathbf\{R\}_\{i\},\\mathbf\{v\}_\{i\},\\mathbf\{p\}_\{i\}"])
    # The group is the one the sentence after the state set declares R_i to live in (M0-F08:
    # this used to be checked against the claim YAML only).
    where = re.search(rf"\$ \({re.escape(rai['eq']['state_set'])}\) where \$\\mathbf\{{R\}}_\{{i\}}"
                      r"\\in\\mathrm\{(\w+)\}\((\d)\)\$ and \$\\mathbf\{v\}_\{i\},\\mathbf\{p\}_\{i\}"
                      r"\\in\\mathbb\{R\}\^\{(\d)\}\$ are the orientation, velocity, and position",
                      source_prose("rai"))
    assert where, f"rai no longer declares the group of R_i after Eq. ({rai['eq']['state_set']})"
    assert rai["group"] == f"{where.group(1)}({where.group(2)})", (rai["group"], where.groups())
    assert rai["velocity_space"] == rai["position_space"] == f"R^{where.group(3)}"
    # No composite group element and no composite tangent: the only matrix Lie group rai ever
    # names is SO(3), and every hatted vector it uses is declared in R^3.
    if rai["element_order"] is None:
        groups = set(re.findall(r"\\math(?:rm|cal|bf)\{(S[OEL]|Gal|GL|U)\}(?:_\{?\d\}?)?\((\d)\)", html))
        assert groups == {("SO", "3")}, groups
    if rai["tangent_order"] is None:
        hatted = set(re.findall(r"\\boldsymbol\{\\(\w+)\}(?:_\{\w+\})?\{\}\^\{\\wedge\}", html))
        declared = {n: d for n, d in re.findall(r"\\boldsymbol\{\\(\w+)\}(?:_\{\w+\})?\\in\\mathbb\{R\}\^\{(\d+)\}", html)}
        assert hatted and all(declared.get(n) == "3" for n in hatted), (hatted, declared)

    # The absent-term scan.  The terms are the claim's.  The statement writes the LaTeX one
    # YAML-escaped ("\\mathrm{SE}"); it is unescaped once so the scan looks for the spelling
    # the HTML would carry -- the first version also counted the doubled spelling, whose count
    # is forced to 0 by the single one (F-7.7).
    stmt = claim("M0-C01")["statement"]
    absent_src = stmt.split("zero occurrences of each of", 1)[1].split("while", 1)[0]
    absent = tuple(dict.fromkeys(t.replace("\\\\", "\\")
                                 for t in re.findall(r'"([^"]+)"', absent_src)))
    for i, term in enumerate(absent):
        for j, other in enumerate(absent):
            assert i == j or term not in other, (
                f"{term!r} is contained in {other!r}; its count cannot fail independently"
            )
    counts = {term: html.count(term) for term in absent}
    assert counts == dict.fromkeys(absent, 0), counts

    present = re.findall(r'"([^"]+)" occurs at least once', stmt)
    assert present, stmt
    for term in present:
        assert html.count(term) >= 1, term
    assert html.count("Mahony") == int(
        claim_number("M0-C01", "test_hint", r'"Mahony"\s*==\s*(\d+)')
    )


# ======================================================================================
# M0-C02
# ======================================================================================

def _rai_rotation_residual_from_source(label: str) -> str:
    """Rai's rotation residual, transcribed mechanically from the source LaTeX into the
    table's alphabet (``R_i``, ``dRbar``, ``xi^``)."""
    s = canon(equation_latex("rai", label))
    s = re.sub(r"^r\^R_ij=", "", s).rstrip(".,")
    s = s.replace("DeltaRbar_ij", " dRbar").replace("xi_ij^", " xi^")
    s = s.replace("R_i", " R_i").replace("R_j", " R_j")
    return s


@pytest.mark.claim("M0-C02")
def test_rai_rotation_residual_signs_and_bias_perturbation():
    assert_claim_statement("M0-C02")          # eq["rotation_residual"] == "19"
    rai = CONVENTIONS["rai"]
    rr = rai["rotation_residual"]
    stmt = claim("M0-C02")["statement"]

    rng = np.random.default_rng(20250219)
    Ri, Rj, dRbar = (Rotation.random(1, rng=rng).as_matrix()[0] for _ in range(3))
    JR = rng.standard_normal((3, 3))
    b_g, bbar_g = rng.standard_normal(3), rng.standard_normal(3)

    # xi_ij as the CLAIM defines it ("xi_ij = J^R_ij * delta_b_g where delta_b_g = b_g - bbar_g")
    sx = re.search(r"xi_ij = (.+?) where (\w+) = (.+?) is the", stmt)
    assert sx, stmt
    s_delta = run_formula(sx.group(3), {"b_g": b_g, "bbar_g": bbar_g})
    s_xi = run_formula(sx.group(1).replace("J^R_ij", "JR"), {"JR": JR, sx.group(2): s_delta})

    # ... and as the TABLE defines it.
    lhs, rhs = (p.strip() for p in rr["xi_definition"].split("=", 1))
    assert lhs == "xi_ij", rr["xi_definition"]
    xi_expr, delta_def = (p.strip() for p in rhs.split(",", 1))
    delta_name, delta_expr = (p.strip() for p in delta_def.split("=", 1))
    assert delta_name == rr["xi_uses"] == sx.group(2), (delta_name, rr["xi_uses"], sx.group(2))
    t_delta = run_formula(delta_expr, {"b_g": b_g, "bbar_g": bbar_g})
    t_xi = run_formula(xi_expr.replace("J^R_ij", "JR"), {"JR": JR, delta_name: t_delta})
    assert np.abs(t_xi - s_xi).max() < 1e-12, (
        f"xi_definition {rr['xi_definition']!r} is not the xi the claim defines"
    )

    # The source decides it too: Eq. (15) builds xi from delta_b_g, and the paper's own
    # sentence defines delta_b_g as the perturbation b_g - bbar_g.
    src15 = canon(equation_latex("rai", rai["eq"]["xi_definition"]))
    m15 = re.search(r"xi_ij=J\^R_ij(\w+)", src15)
    assert m15 and m15.group(1).replace("delta", "delta_") == rr["xi_uses"], (src15, rr["xi_uses"])
    m_using = re.search(r"Using \$([^$]+)\$", source_prose("rai"))
    assert m_using, "the rai source no longer states its delta_b_g definition"
    assert canon_table(delta_def).replace("delta_", "delta") == canon(m_using.group(1)), (
        rr["xi_definition"], m_using.group(1)
    )

    # The recorded terms are EXECUTED and compared with the residual the claim writes out and
    # with Eq. (19) / Eq. (21a) as the paper prints them.  xi comes from the CLAIM's definition,
    # so a corrupted xi_definition cannot hide behind a consistent residual.
    env = {"R_j": Rj, "R_i": Ri, "dRbar": dRbar, "xi": s_xi}
    from_table = sum(c * run_formula(name, env) for c, name in rr["terms"])
    written = re.search(r"r\^R_ij\s*=\s*(.+?)\s*with xi_ij", stmt)
    assert written, "the claim statement no longer spells the residual out"
    claim_form = written.group(1).replace("dRbar_ij", "dRbar").replace("xi_ij^", "xi^")
    for name, text in (("the claim statement", claim_form),
                       ("rai Eq. (19)", _rai_rotation_residual_from_source(rai["eq"]["rotation_residual"])),
                       ("rai Eq. (21a)", _rai_rotation_residual_from_source(
                           rai["eq"]["rotation_residual_restated"]))):
        expected = run_formula(text, env, scalars=("1", "2"))
        assert np.abs(from_table - expected).max() < 1e-12, (
            f"CONVENTIONS['rai']['rotation_residual']['terms'] does not reproduce {name}"
        )


# ======================================================================================
# M0-C03
# ======================================================================================

#: rai symbols, longest first, mapped onto Python names for :func:`run_formula`.
_RAI_SYMBOLS = (("DeltaRbar_ij", "dRbar"), ("Rbreve_ij", "Rbreve"), ("Gamma_ij", "Gamma"),
                ("J^R_ij", "JR"), ("bbar_g", "bbar_g"), ("deltab_g", "delta_b_g"),
                ("b_g", "b_g"), ("Y^'_ij", "Yp"), ("Y_ij", "Y"), ("xi_ij", "xi"),
                ("R_i", "R_i"), ("R_j", "R_j"), ("I_3", "I_3"))


def _rai_expr(text: str) -> str:
    """A canonical rai formula with every symbol spaced out as a Python name, and
    ``( ... )^`` (the hat of a bracketed vector) written as a call."""
    table = dict(_RAI_SYMBOLS)
    text = re.sub("|".join(re.escape(sym) for sym, _ in _RAI_SYMBOLS),
                  lambda m: f" {table[m.group(0)]} ", text)
    prev = None
    while prev != text:
        prev, text = text, re.sub(r"\(([^()]*)\)\^", r" HATOF(\1) ", text)
    return re.sub(r"\s+\^", "^", text)


def _batch_hat(a):
    a = np.asarray(a, dtype=float)
    if a.ndim >= 2 and a.shape[-1] == 1:
        a = a[..., 0]
    return sk(a) if a.ndim == 1 else _skew_batch(a)


def _rai_eval(text: str, env: dict) -> np.ndarray:
    expr = mathify(_rai_expr(text), scalars=("1", "2"), functions=("HATOF",))
    ns = {**env, "HATOF": _batch_hat, "_hat": _batch_hat, "_inv": _inv, "_norm": _norm, "np": np}
    try:
        return np.asarray(eval(expr, {"__builtins__": {}}, ns), dtype=float)   # noqa: S307
    except Exception as exc:
        raise AssertionError(f"rai formula {text!r} does not evaluate ({expr!r}): {exc!r}") from exc


def _rai_definitions(label: str) -> list[tuple[str, str]]:
    """``[(name, rhs)]`` for every ``name = rhs`` definition printed at a rai label."""
    out = []
    for part in canon(equation_latex("rai", label)).rstrip(".,").split(","):
        lhs, rhs = part.split("=", 1)
        out.append((_rai_expr(lhs).strip(), rhs))
    return out


@pytest.mark.claim("M0-C03")
def test_rai_eq22b_is_algebraically_identical_to_eq19():
    """Eq. (22b) is the SAME map as Eq. (19), not a first-order stand-in for it.

    Both sides, and every auxiliary they use (Gamma, Rbreve from (22a); Y, Y' from (22c);
    xi from (15); delta_b_g from the paper's "Using delta_b_g = b_g - bbar_g" sentence),
    are EVALUATED from the source LaTeX printed at the labels the table records (M0-F08:
    the old loop rebuilt both forms from expressions written in the test and read nothing
    from the table).  A corrupted label lands on an equation that does not evaluate, or on a
    different map.  The bar is RELATIVE (F-3): the claim's domain says so.
    """
    rai = CONVENTIONS["rai"]
    eq = rai["eq"]
    bar = claim_number("M0-C03", "statement", r"to (1e-\d+)")
    n = int(claim_number("M0-C03", "statement", r"for (1e\d+) random draws"))
    assert "relative" in claim("M0-C03")["domain"], "the claim's domain must state a relative bar"

    defs = _rai_definitions(eq["lifting_definitions_Gamma"]) + _rai_definitions(eq["lifting_definitions_Y"])
    assert [name for name, _ in defs] == ["Gamma", "Rbreve", "Y", "Yp"], defs
    lifted = canon(equation_latex("rai", eq["lifted_rotation_residual"])).rstrip(".,")
    raw = canon(equation_latex("rai", eq["rotation_residual"])).rstrip(".,")
    assert lifted.startswith("r^R_ij=") and raw.startswith("r^R_ij="), (lifted, raw)
    xi_def = [d for d in canon(equation_latex("rai", eq["xi_definition"])).rstrip(".,").split(",")
              if d.startswith("xi_ij=")]
    assert len(xi_def) == 1, eq["xi_definition"]
    using = re.search(r"Using \$([^$]+)\$", source_prose("rai"))
    assert using, "the rai source no longer states its delta_b_g definition"
    d_lhs, d_rhs = canon(using.group(1)).split("=", 1)
    assert _rai_expr(d_lhs).strip() == "delta_b_g", d_lhs

    rng = np.random.default_rng(20260303)
    for scale in (1.0, 1e-8, 1e4):
        env = {"R_i": Rotation.random(n, rng=rng).as_matrix(),
               "R_j": Rotation.random(n, rng=rng).as_matrix(),
               "dRbar": Rotation.random(n, rng=rng).as_matrix(),
               "JR": rng.standard_normal((n, 3, 3)) * scale,
               "b_g": rng.standard_normal((n, 3, 1)) * scale,
               "bbar_g": rng.standard_normal((n, 3, 1)) * scale,
               "I_3": np.eye(3)}
        for name, rhs in defs:
            env[name] = _rai_eval(rhs, env)
        env["delta_b_g"] = _rai_eval(d_rhs, env)
        env["xi"] = _rai_eval(xi_def[0].split("=", 1)[1], env)
        lhs22b = _rai_eval(lifted.split("=", 1)[1], env)
        lhs19 = _rai_eval(raw.split("=", 1)[1], env)
        assert lhs22b.shape == lhs19.shape == (n, 3, 3), (lhs22b.shape, lhs19.shape)
        magnitude = np.maximum(np.abs(lhs19).max(axis=(1, 2)), 1.0)
        worst = float((np.abs(lhs22b - lhs19).max(axis=(1, 2)) / magnitude).max())
        assert worst < bar, (
            f"rai Eq. ({eq['lifted_rotation_residual']}) is not Eq. ({eq['rotation_residual']}) "
            f"at scale {scale:g}: relative difference {worst:.3e}"
        )


# ======================================================================================
# M0-C04
# ======================================================================================

@pytest.mark.claim("M0-C04")
def test_rai_lifted_vector_and_homogeniser():
    assert_claim_statement("M0-C04", minimum=2)   # eq["lifted_vector"], eq["homogeniser"]
    rai = CONVENTIONS["rai"]
    stmt = claim("M0-C04")["statement"]

    # The block sequence the claim lists, in order, closed by the homogenising scalar.
    scope = {"keyframe": "keyframe", "bias block": "bias_block", "IMU edge": "imu_edge"}
    expected = [
        (scope[kind], name.strip())
        for group, kind in re.findall(r"\[([^]]+)\] per (keyframe|bias block|IMU edge)", stmt)
        for name in group.split(",")
    ]
    last = re.search(r"homogenising scalar (\w+) LAST", stmt)
    assert last, stmt
    recorded = [tuple(x) for x in rai["lifted_vector"]]
    assert [x for x in recorded[:-1]] == expected and recorded[-1][1] == last.group(1), (
        f"lifted_vector records {recorded}; the claim lists {expected} then {last.group(1)!r}"
    )
    # ... and it is Eq. (28)'s own order.
    src28 = canon(equation_latex("rai", rai["eq"]["lifted_vector"]))
    at = -1
    for _, name in recorded:
        pos = src28.find(re.sub(r"\s+", "", name).replace("'", "^'"), at + 1)
        assert pos > at, f"{name!r} is not the next block of rai Eq. (28): {src28}"
        at = pos

    # Eq. (31): every recorded primary constraint is one of the constraints the paper prints
    # there (read symbol-for-symbol off its LaTeX), and all of them are recorded.
    pc = rai["primary_constraints"]
    printed = [p for p in canon(equation_latex("rai", rai["eq"]["homogeniser"])).split(",") if p]
    mine = {k: canon_table(v.split(" over ")[0]) for k, v in pc.items()}
    assert sorted(mine.values()) == sorted(printed), (
        f"primary_constraints {pc} are not rai Eq. (31) {printed}"
    )
    cyc = re.search(r"over cyclic \((\w),(\w),(\w)\)", pc["handedness"])
    assert cyc and "".join(cyc.groups()) == "".join(
        re.findall(r"R_i\^\((\w)\)", printed[[i for i, p in enumerate(printed) if "x" in p][0]])
    ), pc["handedness"]

    # The recorded strings are also RUN at a feasible point of the lifted problem: (R, s) for
    # R in SO(3) and its homogenised mirror (-R, -1), which tests each family's power of s.
    rng = np.random.default_rng(31)
    R = Rotation.random(1, rng=rng).as_matrix()[0]

    def residual(text: str, Rm: np.ndarray, s: float) -> float:
        lhs, rhs = (p.strip() for p in text.split(" over ")[0].split("==", 1))
        env = {"s": float(s), "I_3": np.eye(3)}
        scal = ("s", "1", "2", "3")
        if " x " in lhs:
            worst = 0.0
            for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                left = np.cross(Rm[:, a], Rm[:, b])
                right = run_formula(rhs.replace("R_i^(c)", "C_"), {**env, "C_": Rm[:, c]}, scal)
                worst = max(worst, float(np.abs(left - right).max()))
            return worst
        env["R_i"] = Rm
        return float(np.abs(run_formula(lhs, env, scal) - run_formula(rhs, env, scal)).max())

    for family, text in pc.items():
        for Rm, s in ((R, 1.0), (-R, -1.0)):
            assert residual(text, Rm, s) < 1e-12, (
                f"the recorded {family} constraint {text!r} rejects the feasible point s={s:+g}"
            )


# ======================================================================================
# M0-C05
# ======================================================================================

@pytest.mark.claim("M0-C05")
def test_rai_redundant_constraints_and_ablation():
    assert_claim_statement("M0-C05", minimum=9)   # four cardinalities and five ablation values
    rai = CONVENTIONS["rai"]
    rc = rai["redundant_constraints"]

    mapping = dict(re.findall(r'(\w+)->"(\d+[a-z]?)"', claim("M0-C05")["statement"]))
    assert set(mapping) == set(rc), (set(mapping), set(rc))
    for key, label in mapping.items():
        assert rc[key]["eq"] == label, (key, rc[key]["eq"], label)
        # Table I's id and count are printed inside the equation itself: "(A1, 6/node)".
        src = canon(equation_latex("rai", rc[key]["eq"]))
        tag = re.search(r"\((A1|B1|B2|C1)(?:,([^()]*))?\)\W*$", src)
        assert tag and tag.group(1) == key, (
            f"rai Eq. ({rc[key]['eq']}) is the {tag and tag.group(1)} constraint, not {key}"
        )
        printed = tag.group(2) or COULD_NOT_EXTRACT
        assert rc[key]["cardinality"] == printed, (
            f"{key} cardinality {rc[key]['cardinality']!r}; rai Eq. ({rc[key]['eq']}) prints "
            f"{printed!r}"
        )
        # The recorded identity is the source's chain "L = ... = R", end members compared.
        chain = src[:tag.start()].split("=")
        ident = re.sub(r"\s*\([a-z ]+\)$", "", rc[key]["identity"].split(",  ")[0])
        ident = re.sub(r"\((\w+)\)\^T", r"\1^T", ident.replace("^(')", "")).replace("^.", "^v")
        lhs, rhs = (canon_table(x) for x in ident.split("==", 1))
        assert (lhs, rhs) == (chain[0], chain[-1]), (key, rc[key]["identity"], chain)

    # Sec. VII-B, read off the source text rather than restated.
    ab = rai["ablation"]
    prose = source_prose("rai")
    design = re.search(r"giving \$(\d+)\\times (\d+)\\times (\d+)\\times (\d+)=(\d+)\$ SDP solves",
                       prose)
    assert design, "rai Sec. VII-B no longer states its solve count"
    factors = [int(x) for x in design.groups()[:4]]
    assert ab["total_solves"] == math.prod(factors) == int(design.group(5))
    domain = claim("M0-C05")["domain"]
    assert [int(x) for x in re.findall(
        r"(\d+)\s*(?:primitives|noise levels|seeds|cumulative stages)", domain)] == factors, domain
    gap = re.search(r"through B1, the relative optimality gap remains at least \$([\d.]+)\$ , "
                    r"and none of the corresponding \$(\d+)\$ solves is certified", prose)
    assert gap, "rai Sec. VII-B no longer states the through-B1 result"
    assert ab["gap_through_B1_at_least"] == float(gap.group(1))
    assert ab["certified_through_B1"] == f"0/{gap.group(2)}"
    med = re.search(r"the median gap is \$([\d.]+)\\times 10\^\{(-?\d+)\}\$", prose)
    assert med, "rai Sec. VII-B no longer states the full-set median gap"
    assert ab["median_gap_full_set"] == float(f"{med.group(1)}e{med.group(2)}")
    decisive = re.search(r"Adding (\w+) substantially reduces both the numerical rank", prose)
    assert decisive and ab["decisive_subset"] == decisive.group(1)
    inst = re.search(r"with \$K=(\d+)\$ keyframes and \$M=(\d+)\$ bias blocks, giving \$n=(\d+)\$",
                     prose)
    assert inst, "rai Sec. VII-B no longer states the ablation instance"
    want = "K={},M={},n={}".format(*inst.groups())
    assert want in re.sub(r"\s+", "", ab["instance"]), (ab["instance"], want)
    assert want in re.sub(r"\s+", "", domain), (domain, want)


# ======================================================================================
# M0-C06
# ======================================================================================

@pytest.mark.claim("M0-C06")
def test_rai_isotropic_weighting_is_structurally_forced():
    assert_claim_statement("M0-C06", minimum=2)   # eq["isotropic_weighting"], isotropy_required
    rai = CONVENTIONS["rai"]

    # The identity the table records is the unnumbered display the paper prints before
    # Eq. (23), symbol for symbol -- and the paper's next sentence is what makes the scalar
    # weight forced rather than chosen.  (The first version checked a textbook property of
    # anisotropic quadratic forms that read nothing from the table, F-7.3.)
    reason = rai["isotropy_reason"]
    m = re.search(r"(tau\*\|\|\w+\|\|\^2)\s*==\s*(tau\*\|\|R_i \w+\|\|\^2) for (R_i) in (\S+?):",
                  reason)
    assert m, f"isotropy_reason no longer records the identity it is cited for: {reason!r}"
    prose = source_prose("rai")
    disp = re.search(r"we use \$([^$]*\\tau[^$]*)\$ This equality ([^.]*?) weight\. (\w+) (\w+) "
                     r"([^.]*?)\.", prose)
    assert disp, "rai Sec. IV-B no longer prints the norm identity and its weight sentence"
    printed = canon(disp.group(1))
    assert printed.startswith(canon_table(f"{m.group(1)}=={m.group(2)}")), (reason, printed)
    assert printed.rstrip(".,").endswith(f"{m.group(3)}in{m.group(4)}"), printed

    # M0-F10: the CONCLUSION is decided from the source's own clause, by its polarity, not by
    # substring presence.  The paper's weight clause is a NECESSITY ("requires an isotropic
    # weight" -- not "holds for any weight", not "does not require") and the next sentence
    # draws the scalar precisions as its CONSEQUENCE ("We therefore approximate ... by scalar
    # precisions") -- not as a free choice ("we choose", "for simplicity").  Both together are
    # what makes the scalar weight structurally forced.
    clause = disp.group(2).strip()
    necessity = {"requires an isotropic": True, "holds only for an isotropic": True,
                 "does not require an isotropic": False, "holds for any": False}
    assert clause in necessity, f"rai's weight clause {clause!r} has an undecided polarity"
    consequence = (disp.group(3), disp.group(4)) == ("We", "therefore") and \
        "scalar precisions" in disp.group(5)
    forced = necessity[clause] and consequence
    concl = re.search(r"that equality holds (only for an isotropic|for any) weight -- so the scalar "
                      r"precision is (structurally forced, not a tuning choice|a tuning choice, not "
                      r"structurally forced)$", reason)
    assert concl, f"isotropy_reason states no decidable conclusion: {reason!r}"
    assert (concl.group(1) == "only for an isotropic") is necessity[clause], (concl.group(1), clause)
    assert concl.group(2).startswith("structurally forced") is forced, (concl.group(2), disp.groups())
    assert rai["isotropy_required"] is forced, (rai["isotropy_required"], disp.groups())
    # ... and the products the reason says the device removes are the ones the paper names.
    named = re.search(r"removes the (R_i\^T \w_j) and (R_i\^T \w_j) products", reason)
    printed_products = re.search(r"contain products such as \$([^$]+)\$ and \$([^$]+)\$ \. Rather than "
                                 r"lifting these products directly, we use", prose)
    assert named and printed_products, (reason, printed_products)
    assert [canon_table(x) for x in named.groups()] == [canon(x) for x in printed_products.groups()], (
        named.groups(), printed_products.groups())
    # It is the display immediately preceding Eq. (23), which defines the scalar precisions.
    after = prose[disp.start(5):disp.start(5) + 400]
    eq23 = canon(equation_latex("rai", rai["eq"]["isotropic_weighting"]))
    assert eq23[:20] in canon(after)
    # Eq. (23) makes every weight a SCALAR: tau^-1 = (1/3) tr Sigma, one per residual family,
    # and the identity the table records is written in that same scalar tau.  (M0-F08: the
    # numeric block that used to follow checked |r| == |R r| for R in SO(3) -- a theorem once
    # the regex above has fixed the identity's shape -- so it could not fail and was deleted.)
    weights = re.findall(r"(tau)_(\w),ij\^-1=tfrac13trSigma\^(\w+)_ij", eq23)
    assert [w[1] for w in weights] == ["R", "v", "p"], eq23
    assert {w[0] for w in weights} == {m.group(1).split("*")[0]}, (weights, m.group(1))


# ======================================================================================
# M0-C07
# ======================================================================================

@pytest.mark.claim("M0-C07")
def test_delama_gal3_element_tangent_inverse_adjoint():
    assert_claim_statement("M0-C07", minimum=2)   # element_order, tangent_order
    d = CONVENTIONS["delama"]

    cited = dict(
        re.findall(r'(element_order|tangent_order|inverse|adjoint)[^.]*?at eq "(\d+[a-z]?)"',
                   claim("M0-C07")["statement"])
    )
    assert d["eq"]["element"] == cited["element_order"]
    assert d["eq"]["tangent"] == cited["tangent_order"]
    assert d["eq"]["inverse"] == cited["inverse"]
    assert d["eq"]["adjoint"] == cited["adjoint"]

    # Orderings read straight off delama's own LaTeX at the recorded labels.
    eq3 = equation_latex("delama", d["eq"]["element"])
    assert element_order_from(eq3) == tuple(d["element_order"]), (
        f"delama Eq. ({d['eq']['element']}) orders the element {element_order_from(eq3)}"
    )
    eq4 = equation_latex("delama", d["eq"]["tangent"])
    col, hat = bmatrices(eq4)[0], bmatrices(eq4)[-1]
    assert tuple(c[0] for c in col) == tuple(d["tangent_order"]), (col, d["tangent_order"])
    table_cells = [[canon_table(c) for c in row.split(",")]
                   for row in re.findall(r"\[([^][]*)\]", d["element_matrix"]) if row.strip()]
    assert table_cells == bmatrices(eq3)[-1], (d["element_matrix"], bmatrices(eq3)[-1])

    rng = np.random.default_rng(56)
    A = Rotation.random(8, rng=rng).as_matrix()
    a = rng.standard_normal((8, 3))
    b = rng.standard_normal((8, 3))
    c = rng.standard_normal(8)
    X = np.stack([
        gal_from_rows(d["element_matrix"], {"A": A[k], "a": a[k], "b": b[k], "c": c[k]})
        for k in range(8)
    ])

    # The recorded inverse formula is EXECUTED (F-5: it used to be restated here and never
    # read).  If any sign or slot is wrong, X @ X^-1 stops being I_5.
    lhs_inv, body = (x.strip() for x in d["inverse_formula"].split("=", 1))
    assert canon_table(lhs_inv) == canon(equation_latex("delama", d["eq"]["inverse"])).split("=")[0]
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

    # Which matrix slot is velocity and which is position is decided by the GROUP LAW: only
    # the position slot picks up the Galilean t2*v1 coupling.  The recorded slot names must
    # sit at those columns.
    P = X[0] @ X[1]
    measured = {}
    for idx, col_i in ((1, 3), (2, 4)):
        simple = X[0][0:3, 0:3] @ X[1][0:3, col_i] + X[0][0:3, col_i]
        coupled = float(np.abs(P[0:3, col_i] - simple).max()) > 1e-6
        measured[idx] = "position" if coupled else "velocity"
    assert d["element_slots"][1] == measured[1] and d["element_slots"][2] == measured[2], (
        d["element_slots"], measured
    )
    assert d["element_slots"][0] == "rotation" and d["element_slots"][3] == "time"
    dom = claim("M0-C07")["domain"]
    for slot, name in zip(d["element_slots"][1:3], d["element_order"][1:3]):
        assert re.search(rf"\b{name} (?:is )?the {slot} slot", dom), (slot, name, dom)

    # The tangent coordinates land in the slots the source's hat matrix puts them in.
    assert _slots_from_hat(hat, d["element_order"], d["element_slots"], d["tangent_order"]) == \
        tuple(d["tangent_slots"]), (hat, d["tangent_slots"])

    # The time coordinate is Ad-invariant: the last block row of Ad, computed from the
    # recorded element and tangent layout, must be the recorded row -- and the row the
    # source prints at Eq. (6).
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
    computed = tuple(int(round(float(np.abs(time_row[blocks[s]]).max())))
                     for s in d["tangent_slots"])
    assert d["adjoint_time_row"] == computed, (d["adjoint_time_row"], computed)
    src6 = bmatrices(equation_latex("delama", d["eq"]["adjoint"]))[0][-1]
    assert tuple(0 if cell.startswith("0") else int(cell) for cell in src6) == \
        tuple(d["adjoint_time_row"]), (src6, d["adjoint_time_row"])
    assert d["adjoint_time_invariant"] is (
        all(x == 0 for x in d["adjoint_time_row"][:-1]) and d["adjoint_time_row"][-1] == 1
    )

    # The letters differ between delama and gtsam; the ORDERING does not.
    hint = re.search(r'CONVENTIONS\["gtsam"\]\s*\["tangent_order"\]\s*==\s*(\([^)]*\))',
                     claim("M0-C07")["test_hint"])
    assert hint, claim("M0-C07")["test_hint"]
    assert tuple(CONVENTIONS["gtsam"]["tangent_order"]) == ast.literal_eval(hint.group(1))
    assert d["tangent_slots"] == CONVENTIONS["gtsam"]["tangent_slots"]

    # The constructor tuple is the source's own preintegration tuple.
    tup = re.search(r"preintegration matrix \$\{?\\mathbf\{\\Upsilon\}_\{\\!k\}=\\left\((.*?)\\right\)",
                    source_prose("delama"))
    assert tup, "delama Sec. IV-C no longer prints the preintegration tuple"
    names = tuple(re.sub(r"_k$", "", x).replace("Delta", "d") for x in canon(tup.group(1)).split(","))
    assert tuple(d["ctor_order"]) == names, (d["ctor_order"], names)


# ======================================================================================
# M0-C08
# ======================================================================================

@pytest.mark.claim("M0-C08")
def test_delama_symmetry_group_dim20_and_right_action():
    assert_claim_statement("M0-C08", minimum=5)
    d = CONVENTIONS["delama"]

    # dim == dim(Gal(3)) + dim(gal(3)), consistent with the block shapes of Eqs. (36)/(37).
    assert tuple(name for name, _ in d["group_dim_components"]) == tuple(
        p.strip() for p in d["group"].split("|x")
    ), (d["group_dim_components"], d["group"])
    assert sum(dim for _, dim in d["group_dim_components"]) == d["state_dim"]
    assert sum(d["block_row_dims"]) == d["state_dim"]
    assert d["linearised_error_dim"] == d["state_dim"]
    assert d["A_hat_shape"] == d["B_hat_shape"] == (d["state_dim"], d["state_dim"])
    for label in (d["eq"]["error_dynamics_A"], d["eq"]["error_dynamics_B"]):
        rows = bmatrices(equation_latex("delama", label))[0]
        assert len(rows) == len(d["block_row_dims"]), (label, rows)
        zero = [c for row in rows for c in row if re.fullmatch(r"0_\d+x\d+", c)]
        assert zero and all(c == "0_{0}x{1}".format(*d["block_row_dims"]) for c in zero), (label, zero)

    # The semidirect laws are the ones Eqs. (1)/(2) print, symbol for symbol ...
    for field, key in (("semidirect_product_law", "semidirect_product"),
                       ("semidirect_inverse_law", "semidirect_inverse")):
        printed = canon(equation_latex("delama", d["eq"][key])).rstrip(",.")
        assert canon_table(d[field]) == printed, (field, d[field], printed)

    # ... and they are EXECUTED on the group they describe: (A, a), A in Gal(3), a in gal(3),
    # Ad_A[u] = A u A^-1.  X X^-1 must come out as the identity pair.
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

    def ev(text, env):
        t = re.sub(r"Ad_\{A\^-1\}\[(\w)\]", r"ADINV(\1)", text)
        t = re.sub(r"Ad_A\[(\w)\]", r"ADA(\1)", t)
        t = re.sub(r"\b[A-Za-z]{2,}\b", lambda m: " ".join(m.group()) if (
            m.group() not in env and all(ch in env for ch in m.group())) else m.group(), t)
        return run_formula(t, env, functions=("ADA", "ADINV"))

    A, a = _rand_gal(rng), rand_alg()
    env = {"A": A, "a": a, "ADA": Ad(A), "ADINV": Ad(np.linalg.inv(A))}
    Ai, ai = (ev(p, env) for p in inv_parts)
    idA, ida = (ev(p, {**env, "B": Ai, "b": ai}) for p in prod_parts)
    assert np.abs(idA - np.eye(5)).max() < 1e-9, (
        f"{d['semidirect_product_law']!r} / {d['semidirect_inverse_law']!r} do not compose "
        f"to the identity"
    )
    assert np.abs(np.asarray(ida)).max() < 1e-9

    # The action is a RIGHT action -- the source says so, verbatim, after Eq. (24).
    html = source_html("delama")
    other = {"right": "left", "left": "right"}[d["action_handedness"]]
    assert html.count(f"{d['action_handedness']} group action") >= 1
    assert html.count(f"{other} group action") == 0
    quoted = re.search(r"'([^']+)'", d["action_handedness_citation"])
    assert quoted, d["action_handedness_citation"]
    prose = source_prose("delama")
    eq24 = prose.find(f"$ ({d['eq']['state_action']}) ")
    assert eq24 != -1, f"delama prints no Eq. ({d['eq']['state_action']})"
    following = re.sub(r"\$([^$]*)\$", lambda m: canon(m.group(1)), prose[eq24 + 1:eq24 + 300])
    following = re.sub(r"\s+([.,;:])", r"\1", following).split(" ", 2)[2]
    assert following.startswith(re.sub(r"\s+", " ", quoted.group(1))), (
        f"the sentence after delama Eq. ({d['eq']['state_action']}) is not "
        f"{quoted.group(1)!r}: {following[:160]!r}"
    )
    assert f"Eq. ({d['eq']['state_action']})" in d["action_handedness_citation"]
    assert f" {d['action_handedness']} group action" in quoted.group(1)
    assert_equation_says("delama", d["eq"]["state_action"], require=[r"\\phi\(X_\{k\},\\xi_\{k\}\)"])

    # The symmetry group is NOT the manifold -- both read off the source (M0-F08: they used to
    # be checked against the claim YAML only), and the claim's domain agrees.
    dprose = source_prose("delama")
    sym = re.search(r"symmetry based on the left-trivialized tangent group \$\{?(.+?)\}?\$", dprose)
    assert sym, "delama Sec. I no longer names its symmetry group"
    def _grp(latex):
        latex = re.sub(r"\\mathfrak\{(\w+)\}", r"\1", latex).replace("\\ltimes", " |x ")
        return canon(latex).replace("|x", " |x ")
    assert canon_table(d["group"]).replace("|x", " |x ") == _grp(sym.group(1)), (d["group"], sym.group(1))
    man = re.search(r"Define \$\{?\\mathcal\{M\}\\coloneqq(.+?)\}?\$ and let", dprose)
    assert man, "delama no longer defines its manifold M"
    assert canon_table(d["manifold"]) == canon(man.group(1)), (d["manifold"], man.group(1))
    dom = re.search(r"the manifold is M = (.+?)\.?\s*$", claim("M0-C08")["domain"])
    assert dom and d["manifold"] == dom.group(1), (d["manifold"], claim("M0-C08")["domain"])


# ======================================================================================
# M0-C09
# ======================================================================================

@pytest.mark.claim("M0-C09")
def test_delama_ten_dim_bias_with_four_virtual_channels():
    assert_claim_statement("M0-C09", minimum=5)
    assert_claim_statement("M0-C09", "test_hint", minimum=2)   # gtsam bias_dim, virtual slices
    d, g = CONVENTIONS["delama"], CONVENTIONS["gtsam"]
    prose = source_prose("delama")

    # The bias vector exactly as Sec. IV-C prints it, dimension included.
    m = re.search(r"\\bm\{b\}\}_\{k\}=\\left\((.*?)\\right\)\\in\\mathbb\{R\}\^\{(\d+)\}", prose)
    assert m, "delama Sec. IV-C no longer prints the bias vector"
    names = tuple(re.sub(r"k$", "", x) for x in canon(m.group(1)).split(","))
    assert tuple(d["bias_order"]) == names, (d["bias_order"], names)
    assert d["bias_dim"] == int(m.group(2))

    # The virtual blocks are the TAIL of the recorded order, and their widths are the
    # dimensions the source gives them ("a velocity bias b in R^3 and a time bias b in R").
    virt = re.search(r"two additional virtual biases: a (\w+) bias \$\{?(.*?)\}?\$ and a (\w+) "
                     r"bias \$\{?(.*?)\}?\$", prose)
    assert virt, "delama Sec. IV-C no longer names its two virtual biases"
    widths = []
    for sym in (virt.group(2), virt.group(4)):
        dim = re.search(r"\\in\\mathbb\{R\}(?:\^\{(\d+)\})?", sym)
        widths.append(int(dim.group(1) or 1))
    slices = d["bias_virtual_slices"]
    assert [s.stop - s.start for s in slices] == widths, (slices, widths)
    assert slices[-1].stop == d["bias_dim"] and all(
        lo.stop == hi.start for lo, hi in zip(slices, slices[1:])
    ), slices
    assert d["bias_virtual_names"] == tuple(d["bias_order"][-len(slices):])
    assert d["bias_virtual_kinds"] == tuple(f"virtual {virt.group(k)} bias" for k in (1, 3))

    # Q_d -- the zero process noise -- is on the unnumbered "Define:" line of Algorithm 1;
    # the zero blocks there cover exactly the virtual channels.
    q = re.search(r"Q_\{\\scriptscriptstyle\{d\}\}\}=\\mathrm\{diag\}\((.*?)\)\}?\$", prose)
    assert q, "delama Algorithm 1 no longer defines Q_d"
    halves = canon(q.group(1)).split(",")
    zeros = [int(re.fullmatch(r"0_(\d+)x1", h).group(1)) for h in halves if h.startswith("0_")]
    zero_width = sum(_w for n, _w in zip(d["bias_virtual_names"], widths)
                     if n in d["bias_zero_process_noise"])
    assert zeros and all(z == zero_width for z in zeros), (zeros, d["bias_zero_process_noise"])
    assert d["eq"]["bias_process_noise"] == COULD_NOT_EXTRACT
    # (M0-F08: a scan asserting Q_d sits in no numbered equation was deleted -- it read no
    # table value, so its outcome was fixed by the sha-pinned fixture bytes.)

    # The virtual inputs: Algorithm 1's input line and the claim's domain both fix them.
    inp = re.search(r"Input: \$\{\\prescript\{\}\{\}\{\\tilde\{\\bm\{w\}\}\}_\{k\}=\\left\((.*?)\\right\)",
                    prose)
    assert inp, "delama Algorithm 1 no longer prints its input"
    tail = canon(inp.group(1)).split(",")[-len(slices):]
    for name, printed in zip(d["bias_virtual_names"], tail):
        key = name.split("_", 1)[1] + "_k"
        assert canon_table(d["bias_virtual_inputs"][key]) == printed, (key, printed)
        dom = re.search(rf"{key} = (\S+?)[ ;.]", claim("M0-C09")["domain"])
        assert dom and dom.group(1) == ("0" if printed.startswith("0_") else printed), dom

    # The two bias models cannot be silently interchanged: gtsam's is what the library holds.
    assert g["bias_dim"] == gtsam.imuBias.ConstantBias().vector().size


# ======================================================================================
# M0-C10
# ======================================================================================

#: The equations each row's estimator is built on: the residual / error / minimised cost.
#: Each key is pinned by :data:`EQ_SIGNATURES` to the source's own residual definition or
#: ``\min`` problem (M0-F08: barfoot's used to be the Cayley-map DEFINITION Eq. (14a), a test
#: author's choice; it is now the two Cayley-based costs the paper minimises).  Whether
#: exp/log is *essential* is read off those equations' own LaTeX.
_RESIDUAL_EQ = {
    "rai": ("rotation_residual",),
    "barfoot": ("rotation_averaging_cost", "pose_averaging_cost"),
    "delama": ("error_local_coords",),
    "gtsam": ("residual",),
}
_TRANSCENDENTAL = r"\\log|\\exp|\\operatorname\{Log\}|\\operatorname\{Exp\}|logmap|\\ln"


@pytest.mark.claim("M0-C10")
def test_exp_log_essential_column():
    assert_claim_statement("M0-C10", minimum=2)   # the two delama eq labels
    expected = {k: v == "True" for k, v in
                re.findall(r'"(\w+)":\s*(True|False)', claim("M0-C10")["statement"])}
    assert set(expected) == set(CONVENTIONS), expected

    for key, want in expected.items():
        row = CONVENTIONS[key]
        assert row["exp_log_essential"] == want, (key, row["exp_log_essential"], want)
        for name in _RESIDUAL_EQ[key]:
            latex = equation_latex(key, row["eq"][name])
            found = bool(re.search(_TRANSCENDENTAL, latex))
            assert found == row["exp_log_essential"], (
                f"{key} Eq. ({row['eq'][name]}) "
                f"{'contains' if found else 'does not contain'} a transcendental map"
            )

    d = CONVENTIONS["delama"]
    stmt = claim("M0-C10")["statement"]
    want_eq = dict(re.findall(r'(Gal\(3\) logarithm|inverse left Jacobian J_L\^-1) \(eq "(\d+)"\)',
                              stmt))
    assert len(want_eq) == 2, stmt
    for label, entry in d["exp_log_dependencies"].items():
        assert entry["eq"] == d["eq"][label]
        latex = equation_latex("delama", entry["eq"])
        assert set(entry["requires"]) == set(entry["requires_eq"]) == {"log", "J_L^{-1}"}
        assert re.search(r"\\log\(", latex), label
        assert re.search(r"\\mathbf\{J\}_\{\\mathrm\{L\}\}\(.*\)\^\{-1\}", latex), label
        assert entry["requires_eq"]["log"] == d["eq"]["log"] == want_eq["Gal(3) logarithm"]
        assert entry["requires_eq"]["J_L^{-1}"] == d["eq"]["left_jacobian"] == \
            want_eq["inverse left Jacobian J_L^-1"]

    # Falsifiability anchor for the two False entries (the claim's test_hint): rai removes Log
    # via the chordal residual AND Exp via Cayley.
    assert_claim_statement("M0-C10", "test_hint", minimum=2)
    rai = CONVENTIONS["rai"]
    assert_equation_says("rai", rai["eq"]["chordal_residual"],
                         require=[r"\\mathrm\{Exp\}", r"_\{F\}\^\{2\}"])
    assert_equation_says("rai", rai["eq"]["cayley"], require=[r"\\mathrm\{Cay\}"])


# ======================================================================================
# M0-C11
# ======================================================================================

#: How an error equation composes the true and estimated element, as LaTeX patterns.
_ERROR_FORMS = {
    "X Xh^-1": (r"\\mathbf\{\\Upsilon\}_\{\\!k\}\\hat\{\\mathbf\{\\Upsilon\}\}_\{\\!k\}\^\{-1\}",
                r"\\Upsilon_k\\hat\\Upsilon_k\^\{-1\}",
                # barfoot: cay^-1(T Ttilde_m^-1), cay^-1(C Ctilde_m^T) (C^T = C^-1 on SO(3))
                r"\\mbox\{cay\}\^\{-1\}\\left\(\\mathbf\{T\}\\tilde\{\\mathbf\{T\}\}_\{m\}\^\{-1\}\\right\)",
                r"\\mbox\{cay\}\^\{-1\}\\left\(\\mathbf\{C\}\\tilde\{\\mathbf\{C\}\}_\{m\}\^\{T\}\\right\)"),
    "Xh^-1 X": (r"\\hat\\Upsilon_k\^\{-1\}\\Upsilon_k",
                r"\\hat\{\\mathbf\{\\Upsilon\}\}_\{\\!k\}\^\{-1\}\\mathbf\{\\Upsilon\}_\{\\!k\}"),
}
_ERROR_FN = {
    "X Xh^-1": lambda X, Xh: X @ np.linalg.inv(Xh),
    "Xh^-1 X": lambda X, Xh: np.linalg.inv(Xh) @ X,
}
#: The equations each row's group error / residual is defined by (cf. :data:`_RESIDUAL_EQ`).
_ERROR_EQ = {
    "rai": ("rotation_residual",),
    "delama": ("error_local_coords",),
    "barfoot": ("rotation_averaging_cost", "pose_averaging_cost"),
    "gtsam": ("error",),
}


def _rand_element(key, rng):
    if key == "barfoot":                 # SE(3), Eq. (8)
        T = np.eye(4)
        T[0:3, 0:3] = Rotation.random(1, rng=rng).as_matrix()[0]
        T[0:3, 3] = rng.standard_normal(3)
        return T
    return _rand_gal(rng)


#: The minimised cost each row's error / residual enters, and the weight it is printed with.
#: Only rai prints its cost term with a SCALAR weight (Eq. (20): tau_{R,ij} ||r^R_ij||_F^2);
#: barfoot prints a general W_m inside Eqs. (25)/(28), and delama (34) / gtsam (21) print the
#: error alone, i.e. any quadratic form of it.  A general weight is drawn as a random SPD.
_COST_WEIGHT = {
    "rai": ("inertial_factor",
            r"\\tau_\{R,ij\}\\left\\\|\\mathbf\{r\}\^\{R\}_\{ij\}\\right\\\|_\{F\}\^\{2\}"),
    "barfoot": (None, r"\\mathbf\{W\}_\{m\}"),
}


def error_invariance(key: str, seed: int = 2101) -> dict[str, float]:
    """Relative change of each row's GROUP-VALUED ERROR COMPONENT under left and right translation.

    THE definition of ``error_handedness`` is in the ``src/conventions.py`` module docstring:
    the invariance of the source's group-valued error component, with every non-group quantity
    (R^3 states, bias terms, relative measurements entering as fixed data) held fixed.

    * a printed composition ``X Xh^-1`` / ``Xh^-1 X`` (delama, barfoot, gtsam) is EXECUTED on
      random group elements and enters ``vec(E - I)^T W vec(E - I)`` with the weight the
      source prints (general SPD unless a scalar is printed); X and Xh both translate;
    * rai's Eq. (19) composes no inverse; the residual is EVALUATED from its LaTeX and enters
      the scalar-weighted Frobenius term of Eq. (20).  R_i, R_j translate; dRbar_ij and xi_ij
      are fixed data and are held fixed.
    """
    row = CONVENTIONS[key]
    rng = np.random.default_rng(seed)
    out = {"left": 0.0, "right": 0.0}
    for name in _ERROR_EQ[key]:
        latex = equation_latex(key, row["eq"][name])
        forms = [f for f, pats in _ERROR_FORMS.items() if any(re.search(p, latex) for p in pats)]
        assert len(forms) <= 1, (key, name, forms)
        cost_name, weight_pat = _COST_WEIGHT.get(key, (None, None))
        cost_latex = equation_latex(key, row["eq"][cost_name]) if cost_name else latex
        scalar = False
        if weight_pat:
            assert re.search(weight_pat, cost_latex), (key, cost_name, cost_latex[:200])
            scalar = weight_pat.startswith(r"\\tau")
        if forms:
            err = _ERROR_FN[forms[0]]
            X, Xh, S = (_rand_element(key, rng) for _ in range(3))
            n = X.size
            def resid(A, B):
                return err(A, B) - np.eye(len(A))
            args = {"base": (X, Xh), "left": (S @ X, S @ Xh), "right": (X @ S, Xh @ S)}
        else:
            # no group composition: evaluate the printed residual on the state rotations
            assert key == "rai", f"{key} Eq. ({row['eq'][name]}) composes no group elements"
            lhs, text = canon(latex).rstrip(".,").split("=", 1)
            assert lhs.startswith("r^R_ij"), lhs     # the residual Eq. (20) weights
            Ri, Rj, dRbar, S = (Rotation.random(1, rng=rng).as_matrix()[0] for _ in range(4))
            env = {"dRbar": dRbar, "xi": rng.standard_normal((3, 1)), "I_3": np.eye(3)}
            n = 9
            def resid(A, B):
                return _rai_eval(text, {**env, "R_i": A, "R_j": B})
            args = {"base": (Ri, Rj), "left": (S @ Ri, S @ Rj), "right": (Ri @ S, Rj @ S)}
        if scalar:
            W = rng.uniform(0.5, 2.0) * np.eye(n)
        else:
            L = rng.standard_normal((n, n))
            W = L @ L.T + 0.1 * np.eye(n)
        cost = {k: float(resid(*a).reshape(-1) @ W @ resid(*a).reshape(-1)) for k, a in args.items()}
        for side in ("left", "right"):
            out[side] = max(out[side], abs(cost[side] - cost["base"]) / cost["base"])
    return out


def decide_error_handedness(key: str) -> str:
    """``error_handedness`` decided by its ONE definition (translation invariance of the
    group-valued error component, :func:`error_invariance`) for every row alike."""
    rel = error_invariance(key)
    inv = [side for side, d in rel.items() if d < 1e-9]
    assert len(inv) < 2, (key, rel)          # bi-invariance is not a value of the field
    for side, d in rel.items():
        assert d < 1e-9 or d > 1e-3, (key, side, d)    # no borderline verdicts
    return f"{inv[0]}-invariant" if inv else "none"


#: The bias perturbation each row's correction map is applied to (source symbols).
_BIAS_ARGUMENT = {
    "rai": r"\\mathrm\{Cay\}\(\\boldsymbol\{\\xi\}_\{ij\}\)",
    "delama": r"\\exp\(\(\\mathbf\{J_\{\\Upsilon\}\}_\{k\+1\}\\Delta",
    "gtsam": r"\\operatorname\{Exp\}\(J_k\\delta\\beta\)",
}
_NOMINAL = r"\\hat\{\\mathbf\{\\Upsilon\}\}|\\hat\\Upsilon|\\Delta\\bar\{\\mathbf\{R\}\}"


def decide_bias_correction_side(key: str) -> str:
    """Which side of the nominal pre-integrated quantity the first-order bias correction
    multiplies, read off the source's bias-correction equation; "none" only for a source
    that has no bias at all."""
    row = CONVENTIONS[key]
    if row["bias_order"] is None:
        assert "bias_correction" not in row["eq"], key
        return "none"
    latex = equation_latex(key, row["eq"]["bias_correction"])
    rhs = re.split(r"\\approx|\\simeq|=", latex, maxsplit=1)[1]
    corr = re.search(_BIAS_ARGUMENT[key], rhs)
    nominal = re.search(_NOMINAL, rhs)
    assert corr and nominal, (key, rhs[:200])
    if key == "rai":     # the corrected argument IS the gyro-bias perturbation, Eq. (15) itself
        assert re.search(r"\\boldsymbol\{\\xi\}_\{ij\}\\triangleq\\mathbf\{J\}\^\{R\}_\{ij\}"
                         r"\\delta\\mathbf\{b\}_\{g\}", latex), latex
    return "left" if corr.start() < nominal.start() else "right"


@pytest.mark.claim("M0-C11")
def test_error_handedness_and_bias_correction_side_disagree_on_both_axes():
    assert_claim_statement("M0-C11", minimum=4)
    d, g = CONVENTIONS["delama"], CONVENTIONS["gtsam"]

    # The eq labels the claim cites, per source (the label "34" occurs in BOTH, meaning
    # different equations; each is read only inside its own source's numbering).
    stmt = claim("M0-C11")["statement"]
    halves = stmt.split('CONVENTIONS["gtsam"]')
    cited = {"delama": re.findall(r'eq "(\d+)"', halves[0]),
             "gtsam": re.findall(r'eq "(\d+)"', halves[1])}
    error_eq = {"delama": "error_local_coords", "gtsam": "error"}
    for key in ("delama", "gtsam"):
        row = CONVENTIONS[key]
        assert [row["eq"][error_eq[key]], row["eq"]["bias_correction"]] == cited[key], (
            key, cited[key]
        )
        assert row["error_handedness"] == decide_error_handedness(key), key
        assert row["bias_correction_side"] == decide_bias_correction_side(key), key
    assert d["error_handedness"] != g["error_handedness"]
    assert d["bias_correction_side"] != g["bias_correction_side"]

    # gtsam's left-invariant error goes with a RIGHT perturbation; the recorded retraction is
    # checked against the installed library.
    retr = g["retraction"]
    m = re.search(r"X\s*\(\+\)\s*delta\s*==\s*(.+?)\s*\((\w+) perturbation\)", retr)
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
    assert m.group(2) == ("right" if rhs == "XExp(delta)" else "left"), retr


# ======================================================================================
# M0-C12
# ======================================================================================

def rank_verdict(criterion: dict, M: np.ndarray) -> tuple[float, bool]:
    """Run a recorded numerical-rank rule.  Every ingredient comes out of the table."""
    spectrum, kind = criterion["spectrum"], criterion["kind"]
    if "singular value" in spectrum:
        vals = np.linalg.svd(M, compute_uv=False)
        assert "singular" in kind or "SVR" in kind, (kind, spectrum)
    elif "eigenvalue" in spectrum:
        vals = np.sort(np.abs(np.linalg.eigvalsh(M)))[::-1]
        assert "eigenvalue" in kind, (kind, spectrum)
    else:
        raise AssertionError(f"unknown recorded spectrum {spectrum!r}")

    stat, floor = criterion["statistic"], criterion["relative_floor"]
    if stat.startswith("#{") and "lambda_max" in stat:
        f = re.search(r">\s*([\d.eE+-]+)\s*\*\s*lambda_max", stat)
        assert f and float(f.group(1)) == floor, (stat, floor)
        assert "count" in kind, kind
        value = float((vals > floor * vals.max()).sum())
    elif stat.startswith("log10(") and "/" in stat:
        num, den = (p.strip() for p in stat[len("log10("):-1].split("/"))
        order = {"sigma_1": 0, "sigma_2": 1}
        assert num in order and den in order and num != den, stat
        value = float(np.log10(vals[order[num]] / vals[order[den]]))
        assert math.isclose(floor, 10.0 ** -criterion["threshold"], rel_tol=1e-12), (
            f"a log-ratio rule with threshold {criterion['threshold']} has relative floor "
            f"{10.0 ** -criterion['threshold']:g}, not {floor!r}"
        )
        assert "ratio" in kind or "SVR" in kind, kind
    else:
        raise AssertionError(f"unknown recorded statistic {stat!r}")

    rel = {"==": operator.eq, ">=": operator.ge, ">": operator.gt,
           "<=": operator.le, "<": operator.lt}
    assert criterion["relation"] in rel, criterion["relation"]
    return value, bool(rel[criterion["relation"]](value, criterion["threshold"]))


@pytest.mark.claim("M0-C12")
def test_barfoot_log_svr_and_rai_eigenvalue_rules_are_distinct():
    assert_claim_statement("M0-C12", minimum=2)   # barfoot eq["rank1_criterion"], eq["sparsity"]
    b, r = CONVENTIONS["barfoot"], CONVENTIONS["rai"]
    bc, rc = b["rank1_criterion"], r["rank1_criterion"]
    stmt = claim("M0-C12")["statement"]

    # The claim fixes barfoot's statistic, relation and threshold ...
    thr = re.search(r"(log10\(sigma_1 / sigma_2\))\s*(>=|>)\s*(\d+)", stmt)
    assert thr, stmt
    assert (bc["statistic"], bc["relation"], bc["threshold"]) == (
        thr.group(1), thr.group(2), int(thr.group(3)))
    # ... and barfoot's Sec. 3 prose says the same thing in words.
    bp = source_prose("barfoot")
    svr = re.search(r"we define the logarithmic SVR \( SVR \) as the base-(\d+) logarithm of the "
                    r"ratio of the (\S+) to (\S+) (singular value)s of \$\\mathbf\{X\}\$ ; we "
                    r"consider a log SVR of (at least) \$(\d+)\$ to represent rank 1", bp)
    assert svr, "barfoot Sec. 3 no longer defines the log-SVR"
    rank = {"largest": 1, "second-largest": 2}
    assert bc["statistic"] == "log{}(sigma_{} / sigma_{})".format(
        svr.group(1), rank[svr.group(2)], rank[svr.group(3)]), (bc["statistic"], svr.groups())
    assert bc["relation"] == {"at least": ">="}[svr.group(5)] and bc["threshold"] == int(svr.group(6))
    assert svr.group(4) in bc["spectrum"], bc["spectrum"]

    # rai's rule is the one Sec. VII-A prints: an eigenvalue COUNT at a relative floor.
    floor = claim_number("M0-C12", "statement", r"eigenvalues above ([\d.e-]+) \* lambda_max")
    rp = source_prose("rai")
    eig = re.search(r"number of (eigenvalue)s larger than \$10\^\{(-?\d+)\}\\lambda_\{\\max\}"
                    r"\(\\hat\{\\mathbf\{Z\}\}\)\$ and regard the relaxation as numerically "
                    r"tight when \$\\operatorname\{rank\}\(\\hat\{\\mathbf\{Z\}\}\)(=)(\d+)\$", rp)
    assert eig, "rai Sec. VII-A no longer states its numerical-rank rule"
    assert rc["relative_floor"] == floor == 10.0 ** int(eig.group(2))
    assert rc["relation"] == {"=": "=="}[eig.group(3)], rc["relation"]
    assert rc["threshold"] == int(eig.group(4)), rc["threshold"]
    assert eig.group(1) in rc["spectrum"] and "Z_hat" in rc["statistic"], rc
    assert b["eq"]["rank1_criterion"] == r["eq"]["rank1_criterion"] == COULD_NOT_EXTRACT

    # Numerical anchor in the direction the claim's test_hint states: a spectrum whose
    # second value sits between the two recorded relative floors splits the two rules.
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
    # ... and the two rules agree on unambiguous spectra.
    for eigs, want in (([1.0] + [1e-12] * 9, True), ([1.0, 0.5] + [1e-12] * 8, False)):
        assert rank_verdict(bc, sym(eigs))[1] is want
        assert rank_verdict(rc, sym(eigs))[1] is want

    # Tightness degrades with measurement sparsity (barfoot, Sec. 5) and with stationary
    # segments (rai, Sec. VII-C) -- each source's own words.
    degrades = re.search(r'\["tightness_degrades_with"\]\s*contains "([^"]+)"', stmt)
    assert degrades and degrades.group(1) in b["tightness_degrades_with"], b["tightness_degrades_with"]
    ev = re.search(r"Fig\. (\d+) than Fig\. (\d+) because .*?(start, middle and end)",
                   b["tightness_degrades_evidence"])
    assert ev, b["tightness_degrades_evidence"]
    assert f"the log SVR values are quite a bit lower in Figure {ev.group(1)} as compared to " \
           f"Figure {ev.group(2)} . This seems to be mainly due to the fact that we are now " \
           f"using a sparser set of measurements" in bp, b["tightness_degrades_evidence"]
    assert f"pose measurements only at the {ev.group(3)}" in bp.replace("middle, and", "middle and")
    assert b["eq"]["sparsity"] == COULD_NOT_EXTRACT
    for item in r["tightness_degrades_with"]:
        shape = re.fullmatch(r"(\w+ \w+) \((\w+ \w+)\)", item)
        assert shape, f"rai tightness_degrades_with item {item!r} is not '<cause> (<effect>)'"
        what, why = shape.groups()
        cause = re.search(rf"contains a prolonged {what[:-1]}\w*, leaving the (\w+) (\w+)", rp)
        assert cause and f"{cause.group(2)} {cause.group(1)}" == why, (item, cause)

    # The SDP-side equations the criterion is applied to.
    assert_equation_says("barfoot", b["eq"]["shor_relaxation"],
                         require=[r"\\succeq 0"], forbid=[r"\\mbox\{rank\}"])
    assert_equation_says("barfoot", b["eq"]["rank1_sdp"],
                         require=[r"\\mbox\{rank\}\(\\mathbf\{X\}\)=1"])
    rel = re.fullmatch(r"Shor Eq\. \((\d+[a-z]?)\); tightness == rank\((\w)\) == (\d)", b["relaxation"])
    assert rel and rel.group(1) == b["eq"]["shor_relaxation"], b["relaxation"]
    assert_equation_says("barfoot", b["eq"]["rank1_sdp"],
                         require=[rf"\\mbox\{{rank\}}\(\\mathbf\{{{rel.group(2)}\}}\)={rel.group(3)}"])


# ======================================================================================
# M0-C13
# ======================================================================================

def _ordinal(word: str) -> int:
    return {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4,
            "first": 1, "second": 2, "third": 3, "fourth": 4}[word.lower()]


@pytest.mark.claim("M0-C13")
def test_seven_ordering_incompatibilities_against_installed_gtsam():
    stmt = claim("M0-C13")["statement"]
    assert set(INCOMPATIBILITIES) == set(re.findall(r"\bO[1-7]\b", stmt))
    for key, row in INCOMPATIBILITIES.items():
        assert row["a"]["source"] in CONVENTIONS and row["b"]["source"] in CONVENTIONS, key

    assert _installed_gtsam_version() == CONVENTIONS["gtsam"]["package_version"]
    g, bt, dl = CONVENTIONS["gtsam"], CONVENTIONS["barfoot"], CONVENTIONS["delama"]

    # State dimensions: gtsam's is what the library builds, barfoot's is Eq. (10)'s tangent.
    assert g["state_dim"] == sum(dim for _, dim in g["group_dim_components"]) == (
        gtsam.Gal3.Dim() + gtsam.imuBias.ConstantBias().vector().size
    ), g["state_dim"]
    tan10 = bmatrices(equation_latex("barfoot", bt["eq"]["se3_tangent"]))[0]
    assert tuple(row[0] for row in tan10) == tuple(bt["tangent_order"])
    assert bt["state_dim"] == 3 * len(tan10), (bt["state_dim"], tan10)
    eq8 = element_order_from(equation_latex("barfoot", bt["eq"]["se3_element"]))
    assert eq8 == tuple(bt["element_order"]), (eq8, bt["element_order"])
    for row, key in ((bt, "barfoot"), (g, "gtsam")):
        cells = [[canon_table(c).replace("0_1x3", "0").replace("0^T", "0") for c in r.split(",")]
                 for r in re.findall(r"\[([^][]*)\]", row["element_matrix"]) if r.strip()]
        printed = [[c.replace("0^T", "0") for c in r]
                   for r in bmatrices(equation_latex(key, row["eq"][
                       "se3_element" if key == "barfoot" else "element"]))[-1]]
        assert cells == printed, (key, row["element_matrix"], printed)
    # The group strings are the sets the sources define.
    assert bt["group"] == " and ".join(
        canon(equation_latex("barfoot", bt["eq"][k])).split("=")[0] for k in ("so3", "se3_element"))
    eq1 = canon(equation_latex("gtsam", g["eq"]["state"]))
    assert eq1.startswith("M=" + canon_table(g["group"]) + ","), (g["group"], eq1)
    assert tuple(n for n, _ in g["group_dim_components"]) == tuple(
        x.strip() for x in g["group"].split(" x ")), g["group_dim_components"]
    # rai's bias is the same two IMU 3-blocks as gtsam's, in the other order (O3).
    assert sorted(CONVENTIONS["rai"]["bias_order"]) == sorted(g["bias_order"])
    assert CONVENTIONS["rai"]["bias_dim"] == g["bias_dim"]

    # -- O1: matrix column order vs constructor argument order ---------------------------
    o1 = INCOMPATIBILITIES["O1"]
    assert element_order_from(equation_latex("gtsam", o1["a"]["eq"])) == tuple(g["element_order"])
    ctor_slot = dict(zip(g["ctor_order"], g["ctor_slots"]))
    args = {"rotation": gtsam.Rot3.Identity(), "position": np.array([1.0, 2.0, 3.0]),
            "velocity": np.array([10.0, 20.0, 30.0]), "time": 7.0}
    X = gtsam.Gal3(*[args[ctor_slot[name]] for name in g["ctor_order"]])
    M = X.matrix()
    el_slot = dict(zip(g["element_order"], g["element_slots"]))
    col_of = {el_slot[name]: 3 + i for i, name in enumerate(g["element_order"][1:3])}
    assert set(col_of) == {"velocity", "position"}, g["element_slots"]
    for slot, col in col_of.items():
        assert np.allclose(M[0:3, col], args[slot]), (g["element_order"], slot, col)
    assert M[3, 4] == args["time"]
    assert np.allclose(X.position(), args["position"]) and np.allclose(X.velocity(), args["velocity"])

    for slot in ("velocity", "position"):
        m = re.search(rf"{slot} in (?:matrix )?column index (\d+)", o1["a"]["says"])
        assert m, o1["a"]["says"]
        assert np.allclose(M[0:3, int(m.group(1))], args[slot]), (o1["a"]["says"], slot)
    sig = re.search(r"Gal3\(([^)]*)\)", o1["b"]["says"])
    assert sig and tuple(x.strip() for x in sig.group(1).split(",")) == g["ctor_order"]
    said_args = re.fullmatch(r"Gal3\(([^)]*)\) -- (\w+) is the (\w+) argument, (\w+) the (\w+)",
                             o1["b"]["says"])
    assert said_args, o1["b"]["says"]
    assert {said_args.group(2), said_args.group(4)} == {"position", "velocity"}, o1["b"]["says"]
    for slot, word in (said_args.group(2, 3), said_args.group(4, 5)):
        name = next(n for n, s in ctor_slot.items() if s == slot)
        assert g["ctor_order"].index(name) + 1 == _ordinal(word), o1["b"]["says"]
    assert set(o1["blocks"]) == {g["element_order"][1], g["element_order"][2]}
    rt = o1["runtime_check"]
    m = re.search(r"Gal3\((.*?)\)\.matrix\(\) puts \[([^]]*)\] in column index (\d+) and "
                  r"\[([^]]*)\] in column index (\d+)", rt)
    assert m, rt
    probe = []
    for token in re.findall(r"Rot3\.Identity\(\)|\[[^]]*\]|[-\d.]+", m.group(1)):
        if "Identity" in token:
            probe.append(gtsam.Rot3.Identity())
        elif token.startswith("["):
            probe.append(np.array([float(x) for x in token.strip("[]").split(",")]))
        else:
            probe.append(float(token))
    PM = gtsam.Gal3(*probe).matrix()
    for vec, col in ((m.group(2), m.group(3)), (m.group(4), m.group(5))):
        assert np.allclose(PM[0:3, int(col)], [float(x) for x in vec.split(",")]), (
            f"O1.runtime_check is not what gtsam {_installed_gtsam_version()} does"
        )

    # -- O2: Gal3 tangent vs NavState tangent, and the P_N permutation -------------------
    o2 = INCOMPATIBILITIES["O2"]
    eq5 = equation_latex("gtsam", o2["a"]["eq"])
    assert tuple_after(canon(eq5), "x") == tuple(g["tangent_order"])
    assert _slots_from_hat(bmatrices(eq5)[0], g["element_order"], g["element_slots"],
                           g["tangent_order"]) == tuple(g["tangent_slots"])
    ns_names = tuple(n.replace("delta", "d") for n in
                     tuple_after(canon(equation_latex("gtsam", o2["b"]["eq"])), "n"))
    assert ns_names == tuple(g["navstate_tangent_order"]), (ns_names, g["navstate_tangent_order"])
    blocks = _tangent_blocks(g["tangent_slots"])
    ns_blocks, at = {}, 0
    for slot in g["navstate_tangent_slots"]:
        ns_blocks[slot] = slice(at, at + 3)
        at += 3
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

    # P_N from the doc's own Eq. (38) matrix: which Gal3 block each NavState block row picks.
    assert o2["P_N_eq"] == g["eq"]["navstate_projection"]
    PN = bmatrices(equation_latex("gtsam", o2["P_N_eq"]))[0]
    starts = [blocks[s].start for s in g["tangent_slots"]]
    from_doc = tuple(i for row in PN for j, cell in enumerate(row) if cell == "I_3"
                     for i in range(starts[j], starts[j] + 3))
    ns_args = {"rotation": gtsam.Rot3.Expmap(np.array([0.3, 0.0, 0.0])),
               "position": np.array([1.0, 2.0, 3.0]), "velocity": np.array([4.0, 5.0, 6.0])}
    ns_ctor_slots = {"R": "rotation", "p": "position", "v": "velocity"}
    ns = gtsam.NavState(*[ns_args[ns_ctor_slots[n]] for n in g["navstate_ctor_order"]])
    assert np.allclose(ns.position(), ns_args["position"]) and np.allclose(
        ns.velocity(), ns_args["velocity"]), g["navstate_ctor_order"]
    for slot, blk in ns_blocks.items():
        e9 = np.zeros(9)
        e9[blk] = 0.1
        moved = gtsam.NavState().retract(e9)
        got = {"rotation": np.linalg.norm(moved.attitude().matrix() - np.eye(3)),
               "position": np.linalg.norm(moved.position()),
               "velocity": np.linalg.norm(moved.velocity())}
        assert max(got, key=got.get) == slot, (g["navstate_tangent_slots"], slot, got)
    rows = o2["P_N_rows"]
    assert rows == from_doc == g["navstate_row_selection"], (rows, from_doc)
    spans = re.findall(r"(\d+):(\d+)", o2["P_N_blocks"])
    assert tuple(i for lo, hi in spans for i in range(int(lo), int(hi))) == rows, o2["P_N_blocks"]
    assert g["navstate_block_map"] == tuple(int(lo) for lo, _ in spans)
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
        worst_pn = max(worst_pn,
                       np.linalg.norm(target.localCoordinates(NS.retract(delta[list(rows)]))))
        worst_identity = max(worst_identity,
                             np.linalg.norm(target.localCoordinates(NS.retract(delta[:9]))))
    assert worst_pn < 1e-3 * eps          # second order: the permutation is right
    assert worst_identity > 1e-1 * eps    # first order: dropping it is wrong

    for side, which, slots in (("a", g["tangent_order"], g["tangent_slots"]),
                               ("b", g["navstate_tangent_order"], g["navstate_tangent_slots"])):
        said = re.fullmatch(r"\w+ tangent \(([^)]*)\) -- (\w+) block before (\w+)", o2[side]["says"])
        assert said and tuple(x.strip() for x in said.group(1).split(",")) == which, o2[side]["says"]
        assert slots.index(said.group(2)) < slots.index(said.group(3)), (o2[side]["says"], slots)
    assert set(o2["blocks"]) == {g["tangent_order"][g["tangent_slots"].index(s)]
                                 for s in ("velocity", "position")}, o2["blocks"]

    # -- O3: imuBias::ConstantBias is (accel, gyro) --------------------------------------
    o3 = INCOMPATIBILITIES["O3"]
    assert set(o3["blocks"]) == set(CONVENTIONS["rai"]["bias_order"]) == set(g["bias_order"]), o3["blocks"]
    alias = {"b_omega": "b_g"}          # the doc writes the gyro bias b_omega; the table b_g
    eq1 = bmatrices(equation_latex("gtsam", o3["a"]["eq"]))[-1]
    assert tuple(alias.get(n, n) for n in (c[0].split(",")[0] for c in eq1)) == \
        tuple(g["bias_order"]), (eq1, g["bias_order"])
    probe = {"b_a": np.array([1.0, 2.0, 3.0]), "b_g": np.array([40.0, 50.0, 60.0])}
    cb = gtsam.imuBias.ConstantBias(probe["b_a"], probe["b_g"])
    assert np.allclose(cb.accelerometer(), probe["b_a"]) and np.allclose(cb.gyroscope(), probe["b_g"])
    assert np.allclose(cb.vector(), np.concatenate([probe[n] for n in g["bias_order"]])), (
        f"gtsam bias_order {g['bias_order']} is not the order vector() returns"
    )
    # rai's order is its unnumbered display "b = [b_g^T b_a^T]^T" (between (10c) and (11)).
    disp = re.search(r"Let \$\\mathbf\{b\}\\triangleq\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}",
                     source_prose("rai"))
    assert disp, "rai Sec. III no longer prints its bias vector"
    rai_order = tuple(c.removesuffix("^T") for c in canon(disp.group(1)).split("&"))
    assert tuple(CONVENTIONS["rai"]["bias_order"]) == rai_order, rai_order
    assert CONVENTIONS["rai"]["eq"]["bias_ordering"] == o3["b"]["eq"] == COULD_NOT_EXTRACT
    for side, row_key in (("a", "gtsam"), ("b", "rai")):
        said = re.search(r"\[([^]]*)\]", o3[side]["says"])
        assert said, o3[side]["says"]
        names = tuple(alias.get(x.strip(), x.strip()) for x in said.group(1).split(";"))
        assert names == tuple(CONVENTIONS[row_key]["bias_order"]), (side, names)
    said_delama = re.search(r"delama uses \(([^)]*)\)", o3["b"]["says"])
    assert said_delama and tuple(x.strip() for x in said_delama.group(1).split(",")) == \
        tuple(dl["bias_order"])
    m = re.search(r"ConstantBias\(\[([^]]*)\],\[([^]]*)\]\)\.vector\(\)\s*==\s*\[([^]]*)\]",
                  o3["runtime_check"])
    assert m, o3["runtime_check"]
    acc, gyr, out = (np.array([float(x) for x in grp.split(",")]) for grp in m.groups())
    assert np.allclose(gtsam.imuBias.ConstantBias(acc, gyr).vector(), out), o3["runtime_check"]

    # -- O4: integrateMeasurement takes (acc, omega) -------------------------------------
    o4 = INCOMPATIBILITIES["O4"]
    order = g["imu_sample_order"]
    u10 = [c[0].removesuffix("_k") for c in bmatrices(equation_latex("gtsam", o4["a"]["eq"]))[0]]
    assert len(u10) == len(order) and all(o.startswith(u) for o, u in zip(order, u10)), (u10, order)
    params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    zero = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    sample = {"acc": np.array([1.0, 2.0, 3.0]), "omega": np.zeros(3)}
    pim = gtsam.PreintegratedImuMeasurementsG(params, zero)
    pim.integrateMeasurement(*[sample[k] for k in order], 0.1)
    assert np.abs(pim.deltaRij().matrix() - np.eye(3)).max() < 1e-12
    assert np.allclose(pim.deltaVij(), sample["acc"] * 0.1)
    said = re.fullmatch(r"u = \[([^]]*)\]; integrateMeasurement\(measured(\w+), measured(\w+), "
                        r"deltaT\)", o4["a"]["says"])
    assert said and tuple(x.strip()[0] for x in said.group(1).split(";")) == \
        tuple(s[0] for s in order), o4["a"]["says"]
    assert tuple(x.lower() for x in said.group(2, 3)) == tuple(order), o4["a"]["says"]
    assert set(o4["blocks"]) == set(order), o4["blocks"]
    # rai lists the two channels the other way round -- and rai's own equations say so.
    pair = re.search(r"\((\d+[a-z])\)\((\d+[a-z])\)\s*list\s*(\w+)\s*then\s*(\w+)",
                     o4["b"]["says"])
    assert pair, o4["b"]["says"]
    assert o4["b"]["eq"] == pair.group(1), (o4["b"]["eq"], pair.group(1))
    for label, name in ((pair.group(1), pair.group(3)), (pair.group(2), pair.group(4))):
        lhs = canon(equation_latex("rai", label)).split("=", 1)[0]
        assert lhs == name.replace("_tilde", "tilde") + "(t)", (label, lhs, name)
    m = re.search(r"deltaVij(?:\(\))? == \[([^]]*)\] at dt = ([\d.]+)", o4["runtime_check"])
    assert m, o4["runtime_check"]
    dtv, dt4 = np.array([float(x) for x in m.group(1).split(",")]), float(m.group(2))
    acc4 = re.search(r"integrateMeasurement\(\[([^]]*)\], \[([^]]*)\], dt\)", o4["runtime_check"])
    assert acc4, o4["runtime_check"]
    probe_pim = gtsam.PreintegratedImuMeasurementsG(params, zero)
    probe_pim.integrateMeasurement(np.array([float(x) for x in acc4.group(1).split(",")]),
                                   np.array([float(x) for x in acc4.group(2).split(",")]), dt4)
    assert np.allclose(probe_pim.deltaVij(), dtv), o4["runtime_check"]
    assert np.abs(probe_pim.deltaRij().matrix() - np.eye(3)).max() < 1e-12

    # -- O5: barfoot SE(3) tangent is [rho; phi], translation first ----------------------
    o5 = INCOMPATIBILITIES["O5"]
    assert o5["a"]["eq"] == bt["eq"]["se3_tangent"]
    said = re.search(r"xi = \[([^]]*)\]", o5["a"]["says"])
    assert said and tuple(x.strip() for x in said.group(1).split(";")) == bt["tangent_order"]
    assert o5["blocks"] == bt["tangent_order"]
    hat10 = bmatrices(equation_latex("barfoot", bt["eq"]["se3_tangent"]))[-1]
    assert _slots_from_hat(hat10, bt["element_order"], bt["element_slots"], bt["tangent_order"]) \
        == tuple(bt["tangent_slots"]), (hat10, bt["tangent_slots"])
    lead = re.search(r"--\s*(\w+) first", o5["a"]["says"])
    assert lead and {"translation": "position", "rotation": "rotation"}[lead.group(1)] == \
        bt["tangent_slots"][0], o5["a"]["says"]
    said_b = re.search(r"(.+?) put (\w+) first", o5["b"]["says"])
    assert said_b, o5["b"]["says"]
    for key in [k for k in re.findall(r"\w+", said_b.group(1)) if k in CONVENTIONS]:
        assert CONVENTIONS[key]["tangent_slots"][0] == said_b.group(2)

    # -- O6: the letters r / rho ---------------------------------------------------------
    o6 = INCOMPATIBILITIES["O6"]
    assert g["ctor_order"][g["ctor_slots"].index("position")] == o6["blocks"][0]
    assert bt["tangent_order"][bt["tangent_slots"].index("position")] == o6["blocks"][1]
    assert dl["tangent_order"][dl["tangent_slots"].index("position")] == o6["blocks"][0]
    assert dl["tangent_slots"] == g["tangent_slots"]
    eq4 = bmatrices(equation_latex("delama", o6["b"]["eq"]))[0]
    assert tuple(c[0] for c in eq4) == tuple(dl["tangent_order"])
    six = re.search(r"(\w+) is the (\w+) coordinate; and gtsam's Gal3\(R, (\w+), \.\.\.\) uses "
                    r"(\w+) for (\w+)", o6["a"]["says"])
    assert six, o6["a"]["says"]
    word = {"TRANSLATION": "position", "POSITION": "position"}
    assert bt["tangent_slots"][bt["tangent_order"].index(six.group(1))] == word[six.group(2)]
    assert six.group(3) == six.group(4) == g["ctor_order"][1]
    assert g["ctor_slots"][g["ctor_order"].index(six.group(4))] == word[six.group(5)]
    said6 = re.search(r"delama's \(([^)]*)\) IS gtsam's \(([^)]*)\)", o6["b"]["says"])
    assert said6, o6["b"]["says"]
    assert tuple(x.strip() for x in said6.group(1).split(",")) == dl["tangent_order"]
    assert tuple(x.strip() for x in said6.group(2).split(",")) == g["tangent_order"]

    # -- O7: gravity position-block sign, checked against BOTH sources and by INTEGRATING --
    o7 = INCOMPATIBILITIES["O7"]
    assert (o7["a"]["source"], o7["b"]["source"]) == ("delama", "gtsam")
    assert o7["a"]["eq"] == dl["eq"]["gravity_left_factor"]
    assert o7["b"]["eq"] == g["eq"]["gravity_left_factor"]
    ca, cb_ = o7["a"]["gravity_position_coefficient"], o7["b"]["gravity_position_coefficient"]
    g18 = bmatrices(equation_latex("delama", o7["a"]["eq"]))[0]
    m18 = re.fullmatch(r"([+-]?)\(1/2\)gDeltat_ij\^2", g18[0][2])
    assert m18 and ca == float(f"{m18.group(1)}0.5"), (g18[0][2], ca)
    w15 = re.search(r"W_ij=NavState\(I,([+-]?)\(1/2\)gDeltat_ij\^2,",
                    canon(equation_latex("gtsam", o7["b"]["eq"])))
    assert w15 and cb_ == float(f"{w15.group(1)}0.5"), cb_
    said_w = re.search(r"W_ij = NavState\(I, ([+-]?[\d.]+)\*g\*dt\*\*2, g\*dt\)", o7["b"]["says"])
    assert said_w and float(said_w.group(1)) == cb_, o7["b"]["says"]
    assert canon_table(CONVENTIONS["gtsam"]["endpoint_factors"]["W_ij"]).replace(
        "dt_ij", "dt") == f"NavState(I,{said_w.group(1).lstrip('+')}gdt^2,gdt)", o7["b"]["says"]
    assert o7["blocks"] == ("p",)
    # "applied on the LEFT ... AFTER the coasting map": the factor's place in each endpoint law
    sa = re.fullmatch(r"Gamma_ij = \[\[.*\]\], applied on the (LEFT|RIGHT) of a Gal\(3\) product",
                      o7["a"]["says"])
    sb = re.fullmatch(r"W_ij = NavState\(.*?\), applied on the (LEFT|RIGHT) of an SE_2\(3\) product "
                      r"AFTER the coasting map (\w+)", o7["b"]["says"])
    assert sa and sb, (o7["a"]["says"], o7["b"]["says"])
    d17 = canon(equation_latex("delama", dl["eq"]["endpoint"]))
    assert {"LEFT": d17.startswith("T_j=Gamma_ijT_i"), "RIGHT": d17.endswith("Gamma_ij,")}[sa.group(1)], d17
    g_terms = [t.strip() for t in g["endpoint_relation"].split("=", 1)[1].split(" . ")]
    assert {"LEFT": g_terms[0], "RIGHT": g_terms[-1]}[sb.group(1)] == "W_ij", g_terms
    assert canon_table(g_terms[1]).startswith(sb.group(2)), (g_terms, sb.group(2))

    rng = np.random.default_rng(718)
    worst_p = worst_v = 0.0
    for _ in range(50):
        dt = float(rng.uniform(0.2, 1.5))
        gv = np.array([0.0, 0.0, -9.81])
        Ri = Rotation.random(1, rng=rng).as_matrix()[0]
        vi, pi = rng.standard_normal(3), rng.standard_normal(3)
        dR = Rotation.random(1, rng=rng).as_matrix()[0]
        dv, dp = rng.standard_normal(3), rng.standard_normal(3)

        # the Gal(3) route, with the left factor the table records (and the source prints)
        Gamma = gal_from_rows(o7["a"]["says"], {"g": gv, "dt": dt})
        assert np.array_equal(Gamma, gal_from_latex(g18, {"g": gv, "dt": dt})), (
            "O7.a.says is not delama Eq. (18)"
        )
        Xj = Gamma @ gal(Ri, vi, pi, 0.0) @ gal(dR, dv, dp, dt)
        # the SE_2(3)/NavState route, with gtsam's own coefficient, composed BY GTSAM
        W = gtsam.NavState(gtsam.Rot3.Identity(), cb_ * gv * dt**2, gv * dt)
        phi = gtsam.NavState(gtsam.Rot3(Ri), pi + vi * dt, vi)
        nav = W.compose(phi).compose(gtsam.NavState(gtsam.Rot3(dR), dp, dv))
        worst_p = max(worst_p, float(np.abs(Xj[0:3, 4] - nav.position()).max()))
        worst_v = max(worst_v, float(np.abs(Xj[0:3, 3] - nav.velocity()).max()))
    same_trajectory = worst_p < 1e-9 and worst_v < 1e-9

    # The reason, parsed WHOLE (M0-F06: its last sentence used to be unread; M0-F08: "not a
    # contradiction" used to be a substring test).  Every clause is decided below.
    reason = o7["reason"]
    rm = re.fullmatch(
        r"(?P<verdict>not a contradiction|a contradiction): the sign difference is exactly the "
        r"Galilean (?P<term>\S+) coupling term of the (?P<who>\w+) group product, eq "
        r"(?P<eq>\d+[a-z]?)\. With t2 = (?P<t2>\S+) and v1 = (?P<v1>\S+) the left Gal\(3\) factor "
        r"re-contributes (?P<contrib>[+-][\w*.]+), so the stored entry must be "
        r"(?P<stored>[+-][\w*.]+) to land on (?P<landed>[+-][\w*.]+)\. (?P<grp>\S+) has no such "
        r"coupling because it carries no time coordinate\.", reason)
    assert rm, f"O7.reason no longer has the decidable shape: {reason!r}"
    # verdict: the two coefficients describe the SAME trajectory (computed above, by gtsam)
    assert rm.group("verdict") == ("not a contradiction" if same_trajectory else "a contradiction"), (
        rm.group("verdict"), worst_p, worst_v)
    assert same_trajectory, ("the two recorded gravity coefficients do not describe the same "
                             f"trajectory: {worst_p:.3e} / {worst_v:.3e}")
    assert rm.group("who") == "gtsam" and rm.group("eq") == g["eq"]["product"]
    as_latex = re.sub(r"([a-z])(\d)", r"\1_\2", rm.group("term")).replace("*", "")
    assert_equation_says("gtsam", rm.group("eq"), require=[re.escape(as_latex)])
    gs, dts = 9.81, 0.7
    env = {"g": gs, "dt": dts}
    ev = lambda t: eval(t, {"__builtins__": {}}, env)                          # noqa: E731,S307
    term = ev(f"({rm.group('t2')})*({rm.group('v1')})")
    assert math.isclose(term, ev(rm.group("contrib")))
    stored, landed = ev(rm.group("stored")), ev(rm.group("landed"))
    assert math.isclose(stored + term, landed)
    assert math.isclose(stored, ca * gs * dts**2) and math.isclose(landed, cb_ * gs * dts**2)
    # last sentence: the doc's SE_2(3) is gtsam's NavState (X_i, X_j in SE_2(3) are the
    # NavStates of Eq. (17)); its product has no t2*v1 term and it has no time coordinate.
    assert rm.group("grp") == "SE_2(3)"
    assert r"X_i,X_j\in\mathrm{SE}_2(3)" in source_raw("gtsam")
    A = gtsam.NavState(gtsam.Rot3.Expmap(np.array([0.1, 0.2, 0.3])), np.array([1.0, 2.0, 3.0]),
                       np.array([4.0, 5.0, 6.0]))
    B = gtsam.NavState(gtsam.Rot3.Expmap(np.array([0.3, -0.2, 0.1])), np.array([0.5, 0.1, 0.2]),
                       np.array([0.7, 0.8, 0.9]))
    assert np.abs(A.compose(B).position()
                  - (A.position() + A.attitude().matrix() @ B.position())).max() < 1e-12
    assert A.localCoordinates(B).size == gtsam.Gal3.Dim() - 1
    assert "time" not in g["navstate_tangent_slots"] and "time" in g["tangent_slots"]


@pytest.mark.claim("M0-C13")
def test_incompatibility_consequences_hold_in_installed_gtsam():
    """Each ``consequence`` is parsed whole and decided at runtime or from the source (M0-F06:
    O1, O2, O4 and O6 were never read; O3, O5, O7 neither).  Project policy and pointers to
    this project's own documents were moved to ``commentary`` and are listed in
    UNVERIFIED_COMMENTARY instead of being silently unchecked."""
    from src.conventions import UNVERIFIED_COMMENTARY
    g, dl, bt = CONVENTIONS["gtsam"], CONVENTIONS["delama"], CONVENTIONS["barfoot"]
    O = INCOMPATIBILITIES

    # O1 -- arguments written in MATRIX order (R, v, p, t) are read in constructor order.
    m = re.fullmatch(r"(\w+) and (\w+) swap silently", O["O1"]["consequence"])
    assert m, O["O1"]["consequence"]
    meant = {"velocity": np.array([10.0, 20.0, 30.0]), "position": np.array([1.0, 2.0, 3.0])}
    el_slot = dict(zip(g["element_order"], g["element_slots"]))
    in_matrix_order = [meant[el_slot[n]] if el_slot[n] in meant else None for n in g["element_order"]]
    X = gtsam.Gal3(gtsam.Rot3.Identity(), in_matrix_order[1], in_matrix_order[2], 7.0)   # no error
    got = {"velocity": X.velocity(), "position": X.position()}
    swapped = {s for s in meant if not np.allclose(got[s], meant[s])}
    assert swapped == {m.group(1), m.group(2)}, (swapped, m.groups())
    assert all(np.allclose(got[a], meant[b]) for a, b in (m.groups(), m.groups()[::-1]))

    # O2 -- P_N is a 9-of-10 selection AND a permutation; dropping the permutation swaps blocks.
    m = re.fullmatch(r"the (\d+)-of-(\d+) map is a selection AND a permutation; dropping the "
                     r"permutation silently swaps the (\w+) and (\w+) covariance blocks",
                     O["O2"]["consequence"])
    assert m, O["O2"]["consequence"]
    rows = list(O["O2"]["P_N_rows"])
    assert int(m.group(1)) == len(rows) == gtsam.NavState().localCoordinates(gtsam.NavState()).size
    assert int(m.group(2)) == gtsam.Gal3.Dim() and len(set(rows)) == len(rows) < gtsam.Gal3.Dim()
    assert rows != sorted(rows), "P_N is not a permutation"
    rng = np.random.default_rng(2)
    L = rng.standard_normal((10, 10))
    Sigma = L @ L.T
    right = Sigma[np.ix_(rows, rows)]
    dropped = Sigma[np.ix_(sorted(rows), sorted(rows))]
    ns = {slot: slice(3 * i, 3 * i + 3) for i, slot in enumerate(g["navstate_tangent_slots"])}
    wrong = {slot for slot, blk in ns.items() if not np.allclose(right[blk, blk], dropped[blk, blk])}
    assert wrong == {m.group(3), m.group(4)}, (wrong, m.groups())

    # O3 -- a [b_g; b_a] vector handed to ConstantBias(acc, gyro) puts b_a on the gyroscope.
    m = re.fullmatch(r"the two (\d)-blocks swap: (\w+) bias gets applied to the (\w+)",
                     O["O3"]["consequence"])
    assert m, O["O3"]["consequence"]
    proj = {"b_g": np.array([0.1, 0.2, 0.3]), "b_a": np.array([4.0, 5.0, 6.0])}
    vec = np.concatenate([proj[n] for n in CONVENTIONS["rai"]["bias_order"]])
    w = int(m.group(1))
    cb = gtsam.imuBias.ConstantBias(vec[:w], vec[w:])
    sensor = {"gyroscope": cb.gyroscope(), "accelerometer": cb.accelerometer()}
    which = {"accelerometer": "b_a", "gyroscope": "b_g"}
    assert np.allclose(sensor[m.group(3)], proj[which[m.group(2)]]), (m.groups(), sensor)
    assert w == len(proj["b_g"])

    # O4 -- E (gtsam Eq. 20, R^{10x6}) is exactly the (acc, omega) -> (rotation, velocity) swap.
    m = re.fullmatch(r"the same swap as (O\d) but on the measurements; the lift matrix E \(gtsam "
                     r"Eq\. (\d+), R\^\{(\d+)x(\d+)\}\) is exactly this permutation",
                     O["O4"]["consequence"])
    assert m, O["O4"]["consequence"]
    assert m.group(1) == "O3" and len(O["O3"]["blocks"]) == len(O["O4"]["blocks"]) == 2
    assert m.group(2) == g["eq"]["bias_lift_E"]
    E_src = canon(equation_latex("gtsam", m.group(2)))
    assert f"inR^{m.group(3)}x{m.group(4)}" in E_src, E_src
    cells = bmatrices(equation_latex("gtsam", m.group(2)))[0]
    blocks10 = _tangent_blocks(g["tangent_slots"])
    widths = [blocks10[s].stop - blocks10[s].start for s in g["tangent_slots"]]
    E = np.zeros((int(m.group(3)), int(m.group(4))))
    for r, (row, h) in enumerate(zip(cells, widths)):
        for c, cell in enumerate(row):
            if cell == "I_3":
                E[sum(widths[:r]):sum(widths[:r]) + h, 3 * c:3 * c + 3] = np.eye(3)
    u = {"acc": np.array([1.0, 2.0, 3.0]), "omega": np.array([7.0, 8.0, 9.0])}
    lifted = E @ np.concatenate([u[k] for k in g["imu_sample_order"]])
    assert np.allclose(lifted[blocks10["rotation"]], u["omega"]) and np.allclose(
        lifted[blocks10["velocity"]], u["acc"]), lifted
    assert not lifted[blocks10["position"]].any() and not lifted[blocks10["time"]].any()

    # O5 -- barfoot's tangent order puts a different slot first than every Gal(3) tangent.
    m = re.fullmatch(r"any constraint template copied from (\w+) into a Gal\(3\) lifting lands "
                     r"in the wrong block", O["O5"]["consequence"])
    assert m and m.group(1) == O["O5"]["a"]["source"] == "barfoot", O["O5"]["consequence"]
    for other in (g, dl):
        assert [s for s in other["tangent_slots"] if s in bt["tangent_slots"]] != list(bt["tangent_slots"])

    # O6 -- letters: delama's position letter happens to be gtsam's; letters fix no ordering.
    m = re.fullmatch(r"the letters agree with (\w+) by luck, not by convention\. "
                     r"Do not infer ordering from a letter", O["O6"]["consequence"])
    assert m and m.group(1) == "gtsam", O["O6"]["consequence"]
    pos = lambda row, order: row[order][row[order.replace("order", "slots")].index("position")]  # noqa: E731
    assert pos(dl, "tangent_order") == pos(g, "ctor_order")              # they agree ...
    assert pos(dl, "tangent_order") != pos(g, "tangent_order")           # ... not by a convention
    letters = {}
    for row in (dl, g, bt):
        for i, name in enumerate(row["tangent_order"]):
            letters.setdefault(name, set()).add(i)
    assert any(len(ix) > 1 for ix in letters.values()), letters        # same letter, other index

    # O7 -- +0.5 inside the Gal(3) left factor lands exactly +g*dt^2 off.
    m = re.fullmatch(r"using ([+-][\d.]+)\*g\*dt\*\*2 inside a Gal\(3\) left factor is off by "
                     r"exactly ([+-])g\*dt\*\*2", O["O7"]["consequence"])
    assert m, O["O7"]["consequence"]
    gv, dt = np.array([0.3, -0.2, -9.81]), 0.6
    good = gal_from_rows(O["O7"]["a"]["says"], {"g": gv, "dt": dt})
    bad = gal_from_rows(O["O7"]["a"]["says"].replace(
        f"{O['O7']['a']['gravity_position_coefficient']:+g}*g*dt**2".replace("+", ""),
        f"{float(m.group(1)):+g}*g*dt**2"), {"g": gv, "dt": dt})
    assert not np.array_equal(good, bad), "the consequence names the coefficient already stored"
    X0 = gal(Rotation.random(1, rng=rng).as_matrix()[0], rng.standard_normal(3),
             rng.standard_normal(3), 0.0)
    off = (bad @ X0)[0:3, 4] - (good @ X0)[0:3, 4]
    assert np.allclose(off, float(f"{m.group(2)}1") * gv * dt**2), off

    # commentary is present, marked, and nothing else is
    for path in UNVERIFIED_COMMENTARY:
        node = INCOMPATIBILITIES if path[0] in INCOMPATIBILITIES else CONVENTIONS
        for k in path:
            node = node[k]
        assert isinstance(node, str) and node, path


# ======================================================================================
# M0-C14
# ======================================================================================

@pytest.mark.claim("M0-C14")
def test_gtsam_endpoint_relation_is_W_phi_X_U_and_not_H_X_D():
    assert_claim_statement("M0-C14")      # eq["endpoint"] == "17"
    g = CONVENTIONS["gtsam"]
    stmt = claim("M0-C14")["statement"]

    # The regime is pinned by the claim's domain (F-1): err_wrong is exactly ||v_i||*dt_ij.
    want_dt = claim_number("M0-C14", "domain", r"dt_ij\s*~\s*([\d.]+)")
    want_v = claim_number("M0-C14", "domain", r"\|\|v_i\|\|\s*>\s*([\d.]+)")
    bar_ok = claim_number("M0-C14", "statement", r"to <= (1e-\d+)")
    bar_wrong = claim_number("M0-C14", "statement", r"more than (1e-\d+)")
    # (M0-F08: "want_v * want_dt > bar_wrong" compared two YAML numbers with each other and
    # was deleted; the runtime err_wrong > bar_wrong below is what decides the regime.)

    # The equation labels the claim cites for each factor.
    for factor, field in (("W_ij", "gravity_left_factor"), ("phi_dt", "gravity_left_factor"),
                          ("U_ij", "preintegrated_increment")):
        m = re.search(rf"{re.escape(factor)}(?:\(R,p,v\))? = .*?\(eq \"(\d+)\"\)", stmt)
        assert m and g["eq"][field] == m.group(1), (factor, field, m and m.group(1))

    params, bias, pim = _pim_with_motion(n=50, dt=want_dt / 50)
    dt_ij = pim.deltaTij()
    assert dt_ij == pytest.approx(want_dt, abs=1e-9)
    g_vec = np.asarray(params.n_gravity, dtype=float)
    X_i = gtsam.NavState(gtsam.Rot3.Expmap(np.array([0.11, -0.23, 0.31])),
                         np.array([1.0, 2.0, 3.0]), np.array([0.5, -0.4, 0.3]))
    assert np.linalg.norm(X_i.velocity()) > want_v
    X_j = pim.predict(X_i, bias)

    env = {"NavState": gtsam.NavState, "I": gtsam.Rot3.Identity(), "g": g_vec,
           "dt_ij": float(dt_ij), "dt": float(dt_ij), "dR_ij": pim.deltaRij(),
           "dp_ij": pim.deltaPij(), "dv_ij": pim.deltaVij(),
           "R": X_i.attitude(), "p": X_i.position(), "v": X_i.velocity()}

    def build(text):
        try:
            return eval(text, {"__builtins__": {}}, env)                       # noqa: S307
        except Exception as exc:
            raise AssertionError(f"{text!r} does not evaluate: {exc!r}") from exc

    def nav_equal(a, b):
        return np.linalg.norm(a.localCoordinates(b)) < 1e-12

    # The three factors are BUILT from the strings the table records, and must equal the
    # factors the GTSAM doc prints at Eqs. (15)/(16).
    factors = g["endpoint_factors"]
    W = build(factors["W_ij"])
    U = build(factors["U_ij"])
    phi = gtsam.NavState(*build(factors["phi_dt"].split("=", 1)[1].strip()))
    # (M0-F08: phi is built with gtsam.NavState(...) above, so isinstance(phi, NavState) held
    # by construction and was dropped; W and U are evaluated from the recorded strings.)
    assert all(isinstance(x, gtsam.NavState) for x in (W, U))
    eq15 = canon(equation_latex("gtsam", g["eq"]["gravity_left_factor"]))
    wsrc = re.search(r"W_ij=NavState\(I,([+-]?)\(1/2\)gDeltat_ij\^2,gDeltat_ij\)", eq15)
    assert wsrc, eq15
    W_doc = gtsam.NavState(gtsam.Rot3.Identity(),
                           float(f"{wsrc.group(1)}0.5") * g_vec * dt_ij**2, g_vec * dt_ij)
    assert nav_equal(W, W_doc), (factors["W_ij"], eq15)
    psrc = re.search(r"phi_Deltat\(R,p,v\)=\(R,p\+vDeltat,v\)", eq15)
    assert psrc, eq15
    phi_doc = gtsam.NavState(X_i.attitude(), X_i.position() + X_i.velocity() * dt_ij, X_i.velocity())
    assert nav_equal(phi, phi_doc), (factors["phi_dt"], eq15)
    usrc = canon(equation_latex("gtsam", g["eq"]["preintegrated_increment"]))
    assert usrc.startswith("U_ij(beta_i)=NavState(DeltaR_ij,Deltap_ij,Deltav_ij)"), usrc
    assert nav_equal(U, gtsam.NavState(pim.deltaRij(), pim.deltaPij(), pim.deltaVij()))

    # ... composed in the order the recorded endpoint_relation gives -- Eq. (17)'s order.
    lhs, rhs = (s.strip() for s in g["endpoint_relation"].split("=", 1))
    assert lhs == "X_j", g["endpoint_relation"]
    terms = [t.strip() for t in rhs.split(" . ")]
    built = {"W_ij": W, "U_ij(beta_i)": U, "phi_{dt_ij}(X_i)": phi}
    assert set(terms) == set(built), g["endpoint_relation"]
    eq17 = canon(equation_latex("gtsam", g["eq"]["endpoint"]))
    positions = [eq17.find(canon_table(t).replace("dt", "Deltat")) for t in terms]
    assert -1 not in positions and positions == sorted(positions), (terms, eq17)
    correct = built[terms[0]]
    for t in terms[1:]:
        correct = correct.compose(built[t])
    err_correct = np.linalg.norm(X_j.localCoordinates(correct))
    assert err_correct <= bar_ok, f"{g['endpoint_relation']} misses predict(): {err_correct:.3e}"

    # ... and the coasting-free variant the claim rejects must be visibly WRONG.  The recorded
    # rejected form is parsed WHOLE (M0-F06: only a prefix used to be read): the claim's form,
    # then the name of the factor it drops -- the name the GTSAM doc itself gives that factor
    # in its sentence on the "W phi(X) U separation" (gravity / coasting / IMU motion).
    rejected = re.search(r"NOT the form (.+?);", stmt)
    rf = re.fullmatch(re.escape(rejected.group(1)) + r"  \((\w+) map dropped\)",
                      g["endpoint_rejected_form"]) if rejected else None
    assert rf, g["endpoint_rejected_form"]
    sep = re.search(r"separation is important: (\w+) acts in the navigation frame on the left, "
                    r"autonomous (\w+) advances the existing position, and bias-corrected IMU "
                    r"(\w+) acts in the body/local frame on the right", source_prose("gtsam"))
    assert sep, "the GTSAM doc no longer names the three endpoint factors"
    factor_names = dict(zip(terms, sep.groups()))          # Eq. (17) order: W, phi, U
    kept = ("W_ij", "U_ij(beta_i)")                         # the variant W . X_i . U below
    dropped = [t for t in terms if t not in kept]
    assert len(dropped) == 1 and factor_names[dropped[0]] == rf.group(1), (
        f"the rejected form drops {dropped} = {[factor_names[t] for t in dropped]}, but the "
        f"table says the {rf.group(1)!r} map is dropped"
    )
    err_wrong = np.linalg.norm(X_j.localCoordinates(W.compose(X_i).compose(U)))
    assert err_wrong > bar_wrong, f"W . X_i . U is not distinguishable: {err_wrong:.3e}"
    assert err_wrong == pytest.approx(np.linalg.norm(X_i.velocity()) * dt_ij, rel=1e-6)


# ======================================================================================
# M0-C15
# ======================================================================================

@pytest.mark.claim("M0-C15")
def test_delama_eq18_gamma_equals_T_minus_dt_times_Hprime_exactly():
    """Delama Eq. (18) IS the pre-flight F2 corrected left factor -- bit-identically.

    F-2: the reversed error is exactly ``||g|| dt^2``, so the ``> 1e-6`` bar needs a dt
    floor.  The floor is stated in the claim's domain and read from there.
    """
    o7a = INCOMPATIBILITIES["O7"]["a"]
    assert o7a["eq"] == CONVENTIONS["delama"]["eq"]["gravity_left_factor"]
    g18 = bmatrices(equation_latex("delama", o7a["eq"]))[0]

    # H' and T(-dt) are the forms the CLAIM writes out; Gamma is delama Eq. (18) as the source
    # prints it, and as the table records it.
    stmt = claim("M0-C15")["statement"]
    hp = re.search(r"H' = (\[\[.*?\]\])", stmt)
    tt = re.search(r"T\(-dt\) = (\[\[.*?\]\])", stmt)
    assert hp and tt, stmt
    dt_floor = claim_number("M0-C15", "domain", r"\|dt\|\s*>=\s*([\d.e-]+)")
    g_floor = claim_number("M0-C15", "domain", r"\|\|g\|\|\s*>=\s*([\d.e-]+)")
    bar = claim_number("M0-C15", "test_hint", r"Hp @ T\) > (1e-\d+)")
    # (M0-F08: "g_floor * dt_floor**2 > bar" compared YAML numbers with each other; deleted.)

    rng = np.random.default_rng(1815)
    for _ in range(200):
        dt = float(rng.uniform(dt_floor, 2.0)) * rng.choice([-1.0, 1.0])
        gv = rng.standard_normal(3)
        gv *= rng.uniform(g_floor, 20.0) / np.linalg.norm(gv)
        env = {"g": gv, "dt": dt}

        Gamma = gal_from_latex(g18, env)
        assert np.array_equal(gal_from_rows(o7a["says"], env), Gamma), (
            "INCOMPATIBILITIES['O7']['a']['says'] is not delama Eq. (18)"
        )
        Hp = gal_from_rows(hp.group(1), env)
        T = gal_from_rows(tt.group(1), env)
        assert np.array_equal(Gamma, T @ Hp), f"Eq. (18) is not T(-dt) @ H' at dt={dt:g}"

        # Side placement is load-bearing: the claim's second half, decided against the
        # source's Gamma (M0-F08: this half reads the CLAIM's H', T and the SOURCE, not the
        # table -- it is what a DERIVED claim needs; the table is decided by the line above.
        # The former "== ||g|| dt^2" restatement added nothing a mutation could move: deleted.)
        reversed_err = np.linalg.norm(Gamma - Hp @ T)
        assert reversed_err > bar, f"reversed product indistinguishable at dt={dt:g}"


# ======================================================================================
# Every equation label, row by row, against the source it cites
# ======================================================================================

#: ``path -> (patterns the cited equation must contain, patterns it must not)``.
#: The label looked up is the one the table records, so a corrupted label lands on an
#: equation that fails its signature.  Patterns are on the source's own LaTeX.
EQ_SIGNATURES: dict[tuple, tuple[tuple[str, ...], tuple[str, ...]]] = {
    ("rai", "eq", "cayley"): ((r"\\mathrm\{Cay\}\(", r"2\\tan\(\\theta/2\)"), ()),
    ("rai", "eq", "cayley_right_jacobian"): ((r"\\mathbf\{J\}_\{\\mathrm\{Cay\}\}",), ()),
    ("rai", "eq", "qcqp"): ((r"f_\{\\mathrm\{QCQP\}\}", r"\\mathbf\{A\}_\{j\}"), ()),
    ("rai", "eq", "sdp"): ((r"f_\{\\mathrm\{SDP\}\}", r"\\mathbb\{S\}\^\{n\}_\{\+\}"), ()),
    ("rai", "eq", "chordal_residual"): ((r"\\mathrm\{Exp\}", r"_\{F\}\^\{2\}"), ()),
    ("rai", "eq", "xi_definition"): ((r"\\boldsymbol\{\\xi\}_\{ij\}\\triangleq",), ()),
    ("rai", "eq", "bias_correction"): (
        (r"^\\mathbf\{R\}_\{i\}\^\{\\top\}\\mathbf\{R\}_\{j\}=\\Delta\\bar\{\\mathbf\{R\}\}_\{ij\}\\,"
         r"\\mathrm\{Cay\}\(\\boldsymbol\{\\xi\}_\{ij\}\)",), ()),
    ("rai", "eq", "lifting_definitions_Gamma"): (
        (r"^\\boldsymbol\{\\Gamma\}_\{ij\}\\triangleq", r"\\breve\{\\mathbf\{R\}\}_\{ij\}\\triangleq"), ()),
    ("rai", "eq", "cayley_determinant"): ((r"^\\det", r"1\+\\tfrac\{1\}\{4\}"), ()),
    ("rai", "eq", "denominator_clearing"): (
        (r"^\\mathbf\{R\}_\{j\}\\bigl\(\\mathbf\{I\}_\{3\}-\\tfrac\{1\}\{2\}",), ()),
    ("rai", "eq", "rotation_residual"): ((r"\\mathbf\{r\}\^\{R\}_\{ij\}\\triangleq",), ()),
    ("rai", "eq", "rotation_residual_restated"): ((r"^\\mathbf\{r\}\^\{R\}_\{ij\}=\\left\(",), ()),
    ("rai", "eq", "velocity_residual"): (
        (r"\\mathbf\{r\}\^\{v\}_\{ij\}", r"\\mathbf\{A\}\^\{vg\}_\{ij\}"), ()),
    ("rai", "eq", "position_residual"): (
        (r"\\mathbf\{r\}\^\{p\}_\{ij\}", r"\\mathbf\{A\}\^\{pg\}_\{ij\}"), ()),
    ("rai", "eq", "lifted_rotation_residual"): ((r"\\breve\{\\mathbf\{R\}\}_\{ij\}-",), ()),
    ("rai", "eq", "lifting_definitions_Y"): ((r"\\mathbf\{Y\}\^\{\\prime\}_\{ij\}\\triangleq",), ()),
    ("rai", "eq", "isotropic_weighting"): ((r"^\\tau_\{R,ij\}\^\{-1\}=",), ()),
    ("rai", "eq", "inertial_factor"): (
        (r"^F\^\{\\mathrm\{Cay\}\}_\{ij\}=\\tau_\{R,ij\}\\left\\\|\\mathbf\{r\}\^\{R\}_\{ij\}",), ()),
    # BOTH halves (M0-F06: "26a", the velocity half alone, used to pass).
    ("rai", "eq", "u_definitions"): (
        (r"\\mathbf\{W\}\^\{v\}_\{ij\} \\displaystyle\\triangleq",
         r"\\mathbf\{u\}\^\{v\}_\{ij\} \\displaystyle\\triangleq",
         r"\\mathbf\{W\}\^\{p\}_\{ij\} \\displaystyle\\triangleq",
         r"\\mathbf\{u\}\^\{p\}_\{ij\} \\displaystyle\\triangleq"), ()),
    ("rai", "eq", "lifted_vector"): (
        (r"^\\displaystyle\\mathbf\{x\}\\triangleq", r"\\operatorname\{vec\}\(\\mathbf\{R\}_\{i\}\)"), ()),
    ("rai", "eq", "lifting_constraints"): (
        (r"\\right\)\^\{\\top\}\\mathbf\{Y\}_\{ij\}", r"\{\}\^\{\\wedge\}s"), ()),
    ("rai", "eq", "homogeniser"): ((r"^s\^\{2\}=1",), ()),
    ("rai", "eq", "state_set"): ((r"^\\mathcal\{X\}\\triangleq",), ()),
    ("rai", "redundant_constraints", "A1", "eq"): ((r"\(\\mathrm\{A1\}",), ()),
    ("rai", "redundant_constraints", "B1", "eq"): ((r"\(\\mathrm\{B1\}",), ()),
    ("rai", "redundant_constraints", "B2", "eq"): ((r"\(\\mathrm\{B2\}\)",), ()),
    ("rai", "redundant_constraints", "C1", "eq"): ((r"\(\\mathrm\{C1\}",), ()),

    ("delama", "eq", "semidirect_product"): ((r"XY=", r"\\mathrm\{Ad\}_\{A\}"), ()),
    ("delama", "eq", "semidirect_inverse"): (
        (r"\{X\}\^\{-1\}=", r"-\\mathrm\{Ad\}_\{\{A\}\^\{-1\}\}"), ()),
    ("delama", "eq", "element"): ((r"^\\mathbf\{X\}=\\begin\{bmatrix\}", r"\\mathbb\{R\}\^\{5\\times 5\}"), ()),
    ("delama", "eq", "tangent"): ((r"\\bm\{x\}\}\^\{\\wedge\}=", r"\\alpha"), ()),
    ("delama", "eq", "inverse"): ((r"^\{\\mathbf\{X\}\}\^\{-1\}=", r"\\mathbf\{A\}\^\{\\top\}"), ()),
    ("delama", "eq", "adjoint"): ((r"\\mathbf\{Ad\}\^\{\\vee\}_\{\\mathbf\{X\}\}\\coloneqq",), ()),
    ("delama", "eq", "exp"): ((r"\\exp\(\\prescript\{\}\{\}\{\\bm\{x\}\}\^\{\\wedge\}\)",), ()),
    ("delama", "eq", "log"): ((r"\\log\(X\)",), ()),
    ("delama", "eq", "left_jacobian"): (
        (r"\\mathbf\{J\}_\{\\mathrm\{L\}\}\(\\prescript\{\}\{\}\{\\bm\{x\}\}\)", r"\\mathbf\{Q\}_\{1\}"), ()),
    ("delama", "eq", "endpoint"): ((r"\\mathbf\{T\}\}_\{j\}=\\mathbf\{\\Gamma\}_\{\\!ij\}",), ()),
    ("delama", "eq", "gravity_left_factor"): ((r"\\mathbf\{\\Gamma\}_\{\\!ij\}\\coloneqq",), ()),
    ("delama", "eq", "preintegrated_increment"): (
        (r"\\mathbf\{\\Upsilon\}_\{\\!ij\}\\coloneqq", r"\\Delta\\!\\prescript\{\}\{\}\{\\mathbf\{R\}\}"), ()),
    ("delama", "eq", "mean_propagation"): (
        (r"\\mathbf\{\\Upsilon\}_\{\\!k\+1\}&=\\mathbf\{\\Upsilon\}_\{\\!k\}\\exp",), ()),
    ("delama", "eq", "state_action"): ((r"^\\phi\(X_\{k\},\\xi_\{k\}\)\\coloneqq",), ()),
    ("delama", "eq", "input_action"): ((r"^\\psi\(X_\{k\},u_\{k\}\)\\coloneqq",), ()),
    ("delama", "eq", "equivariant_error"): ((r"^e_\{k\}\\coloneqq\\phi\(\\hat\{X\}_\{k\}\^\{-1\}",), ()),
    ("delama", "eq", "local_chart"): ((r"^\\vartheta\(e_\{k\}\)\\coloneqq\\log_\{G\}",), ()),
    ("delama", "eq", "log_G"): ((r"^\\log_\{G\}\\!\\left\(X_\{k\}\\right\)=",), ()),
    ("delama", "eq", "error_local_coords"): (
        (r"\\bm\{\\varepsilon\}\}_\{k\}\\!=", r"\\mathbf\{\\Upsilon\}_\{\\!k\}\\hat\{\\mathbf\{\\Upsilon\}\}"), ()),
    ("delama", "eq", "error_dynamics"): (
        (r"\\bm\{\\varepsilon\}\}_\{k\+1\}\\approx\\hat\{\\mathbf\{A\}\}",), ()),
    ("delama", "eq", "error_dynamics_A"): ((r"\\hat\{\\mathbf\{A\}\}_\{k\+1\}&=",), ()),
    ("delama", "eq", "error_dynamics_B"): ((r"^\\displaystyle\\hat\{\\mathbf\{B\}\}_\{k\+1\} ",), ()),
    ("delama", "eq", "bias_correction"): (
        (r"\\hat\{\\mathbf\{\\Upsilon\}\}_\{\\!k\+1\}\^\{\+\}\\approx\\exp",), ()),
    ("delama", "exp_log_dependencies", "log_G", "eq"): ((r"^\\log_\{G\}",), ()),
    ("delama", "exp_log_dependencies", "log_G", "requires_eq", "log"): ((r"\\log\(X\)",), ()),
    ("delama", "exp_log_dependencies", "log_G", "requires_eq", "J_L^{-1}"): (
        (r"\\mathbf\{J\}_\{\\mathrm\{L\}\}\(\\prescript",), ()),
    ("delama", "exp_log_dependencies", "error_local_coords", "eq"): (
        (r"\\bm\{\\varepsilon\}\}_\{k\}\\!=",), ()),
    ("delama", "exp_log_dependencies", "error_local_coords", "requires_eq", "log"): (
        (r"\\log\(X\)",), ()),
    ("delama", "exp_log_dependencies", "error_local_coords", "requires_eq", "J_L^{-1}"): (
        (r"\\mathbf\{J\}_\{\\mathrm\{L\}\}\(\\prescript",), ()),

    ("barfoot", "eq", "so3"): ((r"^SO\(3\)=", r"\\mbox\{det\}"), ()),
    ("barfoot", "eq", "se3_element"): ((r"^SE\(3\)=", r"\\mathbf\{C\}&\\mathbf\{r\}"), ()),
    ("barfoot", "eq", "se3_tangent"): ((r"^\{\\boldsymbol\{\\xi\}\}\^\{\\wedge\}=",), ()),
    ("barfoot", "eq", "adjoint"): ((r"\\mbox\{Ad\}\\left\(\\mathbf\{T\}\\right\)",), ()),
    ("barfoot", "eq", "cayley"): (
        (r"\\mathbf\{C\}\(\{\\boldsymbol\{\\phi\}\}\)", r"\\mbox\{cay\}\(\{\\boldsymbol\{\\phi\}\}\^\{\\wedge\}\)"),
        (r"\\mbox\{cay\}\^\{-1\}",)),
    ("barfoot", "eq", "cayley_inverse"): ((r"\\mbox\{cay\}\^\{-1\}\(\\mathbf\{C\}\)",), ()),
    ("barfoot", "eq", "cayley_se3"): ((r"\\mbox\{cay\}\(\{\\boldsymbol\{\\xi\}\}\^\{\\wedge\}\)",), ()),
    ("barfoot", "eq", "nonconvex"): ((r"f\(\\mathbf\{z\}\)", r"g_\{i\}\(\\mathbf\{z\}\)=0"), ()),
    ("barfoot", "eq", "homogeneous_qcqp"): (
        (r"\\mathbf\{x\}\^\{T\}\\mathbf\{A\}_\{0\}\\mathbf\{x\}=1",), (r"\\succeq",)),
    ("barfoot", "eq", "rotation_averaging_cost"): (
        (r"\$\\min\$", r"\\mbox\{cay\}\^\{-1\}\\left\(\\mathbf\{C\}\\tilde\{\\mathbf\{C\}\}_\{m\}\^\{T\}\\right\)",
         r"\\mathbf\{C\}\\in SO\(3\)"), ()),
    ("barfoot", "eq", "pose_averaging_cost"): (
        (r"\$\\min\$", r"\\mbox\{cay\}\^\{-1\}\\left\(\\mathbf\{T\}\\tilde\{\\mathbf\{T\}\}_\{m\}\^\{-1\}\\right\)",
         r"\\mathbf\{T\}\\in SE\(3\)"), ()),
    ("barfoot", "eq", "rank1_sdp"): ((r"\\mbox\{rank\}\(\\mathbf\{X\}\)=1",), ()),
    ("barfoot", "eq", "shor_relaxation"): ((r"\\mathbf\{X\}\\succeq 0",), (r"\\mbox\{rank\}",)),

    ("gtsam", "eq", "state"): ((r"^\\mathcal M=\\mathrm\{Gal\}\(3\)\\times\\mathbb R\^6", r"b_\{a,k\}"), ()),
    ("gtsam", "eq", "element"): ((r"^X=\(R,v,p,t\)",), ()),
    ("gtsam", "eq", "product"): ((r"p_1\+R_1p_2\+t_2v_1",), ()),
    ("gtsam", "eq", "inverse"): ((r"\^\{-1\}=", r"-R\^\\top\(p-tv\)"), ()),
    ("gtsam", "eq", "tangent"): ((r"^x=\(\\omega,\\nu,\\rho,\\alpha\)",), ()),
    ("gtsam", "eq", "exp"): ((r"^\\operatorname\{Exp\}\(x\)=", r"\\alpha\\Gamma_2\(\\omega\)\\nu"), ()),
    ("gtsam", "eq", "adjoint"): ((r"^\\operatorname\{Ad\}_X=", r"\(p-tv\)\^\\wedge R"), ()),
    ("gtsam", "eq", "imu_sample"): ((r"^u_k=",), ()),
    ("gtsam", "eq", "composition"): ((r"^F_h\(\(\\Upsilon_k,\\beta_k\),u_k\)",), ()),
    ("gtsam", "eq", "held_input_composition"): (
        (r"\\hat\\Upsilon_\{i,k\+1\}=\\hat\\Upsilon_\{ik\}V_k",), ()),
    ("gtsam", "eq", "gravity_left_factor"): ((r"^W_\{ij\}=\\operatorname\{NavState\}",), ()),
    ("gtsam", "eq", "preintegrated_increment"): (
        (r"^U_\{ij\}\(\\beta_i\)=\\operatorname\{NavState\}",), ()),
    ("gtsam", "eq", "endpoint"): ((r"X_j=W_\{ij\}", r"\\phi_\{\\Delta t_\{ij\}\}\(X_i\)"),
                                  (r"\\mathcal P_\\Omega",)),
    ("gtsam", "eq", "bias_lift_E"): ((r"^E=", r"\\mathbb R\^\{10\\times6\}"), ()),
    ("gtsam", "eq", "error"): ((r"\\mathsf e_k=\\hat\\xi_k\^\{-1\}\\xi_k",), ()),
    ("gtsam", "eq", "error_vector"): ((r"^\\epsilon_k=\(e_k,\\delta\\beta_k\)",), ()),
    ("gtsam", "eq", "C_matrix"): ((r"C=J_R\(x\)hE",), ()),
    ("gtsam", "eq", "transition_first_order"): ((r"^V_\{\\mathrm\{true\}\}", r"\\simeq"), ()),
    ("gtsam", "eq", "transition"): ((r"A_\{\\mathrm\{LI\}\}", r"\\operatorname\{Ad\}_\{V\^\{-1\}\}"), ()),
    ("gtsam", "eq", "bias_correction"): ((r"^\\Upsilon_k\(\\hat\\beta\+\\delta\\beta\)",), ()),
    ("gtsam", "eq", "bias_jacobian_recursion"): (
        (r"J_\{k\+1\}=\\operatorname\{Ad\}_\{V_k\^\{-1\}\}J_k-C_k",), ()),
    ("gtsam", "eq", "bias_correction_restated"): ((r"^\\Upsilon_k\(\\beta\)\\simeq",), ()),
    ("gtsam", "eq", "navstate_tangent"): ((r"n=\(\\delta\\theta,\\delta p,\\delta v\)",), ()),
    ("gtsam", "eq", "navstate_projection"): ((r"P_N=", r"\\mathbb R\^\{9\\times10\}"), ()),
    ("gtsam", "eq", "residual"): ((r"^r_\{ij\}=\\operatorname\{Local\}", r"logmap"), ()),
    ("gtsam", "eq", "rotating_frame"): ((r"W\^\\Omega_\{ij\}", r"\\mathcal L_\\Omega\(X_i\)"), ()),

    # The incompatibility register's labels: each must be the equation that carries the
    # ordering the row is about.
    ("O1", "a", "eq"): ((r"R & v & p\\\\ 0 & 1 & t",), ()),
    ("O2", "a", "eq"): ((r"^x=\(\\omega,\\nu,\\rho,\\alpha\)",), ()),
    ("O2", "b", "eq"): ((r"n=\(\\delta\\theta,\\delta p,\\delta v\)",), ()),
    ("O2", "P_N_eq"): ((r"P_N=",), ()),
    ("O3", "a", "eq"): ((r"\\beta_k=\\begin\{bmatrix\}b_\{a,k\}\\\\b_\{\\omega,k\}",), ()),
    ("O4", "a", "eq"): ((r"^u_k=\\begin\{bmatrix\}a_k\\\\\\omega_k",), ()),
    ("O4", "b", "eq"): ((r"^\\displaystyle\\tilde\{\\boldsymbol\{\\omega\}\}\(t\)",), ()),
    ("O5", "a", "eq"): ((r"\{\\boldsymbol\{\\rho\}\}\\\\ \{\\boldsymbol\{\\phi\}\}",), ()),
    ("O5", "b", "eq"): ((r"^x=\(\\omega,",), ()),
    ("O6", "a", "eq"): ((r"\{\\boldsymbol\{\\rho\}\}\\\\ \{\\boldsymbol\{\\phi\}\}",), ()),
    ("O6", "b", "eq"): ((r"\\bm\{v\}\}\\\\ \\prescript\{\}\{\}\{\\bm\{r\}\}",), ()),
    ("O7", "a", "eq"): ((r"\\mathbf\{\\Gamma\}_\{\\!ij\}\\coloneqq",), ()),
    ("O7", "b", "eq"): ((r"^W_\{ij\}=\\operatorname\{NavState\}",), ()),
}

#: Entries recorded as COULD_NOT_EXTRACT, with the source prose that states the fact.  Where
#: the fact has a value in the table, the pattern is BUILT from that value, so the prose
#: check reads the table.  (M0-F08: the old second half -- "the fact sits in no numbered
#: equation" -- read no table value; its outcome was fixed by the sha-pinned fixture bytes,
#: and for continuous_time_ins it was vacuous.  It was deleted.  What decides the field is
#: ``value == COULD_NOT_EXTRACT``, which every digit-bearing mutant fails.)
SENTINEL_EVIDENCE: dict[tuple, object] = {
    ("rai", "eq", "bias_ordering"): lambda: (
        r"Let \$\\mathbf\{b\}\\triangleq\\begin\{bmatrix\}" + "&".join(
            rf"\\mathbf\{{b\}}_\{{{n.split('_', 1)[1]}\}}\^\{{\\top\}}"
            for n in CONVENTIONS["rai"]["bias_order"]) + r"\\end\{bmatrix\}"),
    ("rai", "eq", "rank1_criterion"): lambda: (
        r"number of eigenvalues larger than \$10\^\{%d\}\\lambda_\{\\max\}"
        % round(math.log10(CONVENTIONS["rai"]["rank1_criterion"]["relative_floor"]))),
    ("delama", "eq", "bias_process_noise"): lambda: (
        r"Define: .{0,600}\\mathbf\{Q_\{\\scriptscriptstyle\{d\}\}\}=\\mathrm\{diag\}\(.*?"
        r"\\mathbf\{0\}_\{%d\\times 1\}" % sum(
            sl.stop - sl.start for n, sl in zip(CONVENTIONS["delama"]["bias_virtual_names"],
                                                CONVENTIONS["delama"]["bias_virtual_slices"])
            if n in CONVENTIONS["delama"]["bias_zero_process_noise"])),
    ("delama", "eq", "continuous_time_ins"): lambda: r"by extending \( IV-A \) with additional virtual",
    ("barfoot", "eq", "rank1_criterion"): lambda: (
        r"we consider a log SVR of at least \$%d\$ to represent rank 1"
        % CONVENTIONS["barfoot"]["rank1_criterion"]["threshold"]),
    ("barfoot", "eq", "sparsity"): lambda: r"we are now using a sparser set of measurements",
    ("barfoot", "eq", "licq_failure"): lambda: r"we do not meet the technical condition of LICQ",
}


def _walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, path + (key,))
    else:
        yield path, node


def _is_eq_path(path) -> bool:
    return path[-1] in ("eq", "P_N_eq") or path[0] == "eq" or "requires_eq" in path


def _source_of(path) -> str:
    if path[0] in CONVENTIONS:
        return path[0]
    row = INCOMPATIBILITIES[path[0]]
    return row[path[1]]["source"] if path[1] in ("a", "b") else row["a"]["source"]


def _check_eq_entries(prefix: str, root: dict) -> None:
    checked = set()
    for path, (require, forbid) in EQ_SIGNATURES.items():
        if path[0] != prefix:
            continue
        value = table_value(path)
        key = _source_of(path)
        assert value != COULD_NOT_EXTRACT, (
            f"{list(path)} is recorded as {COULD_NOT_EXTRACT!r}, but {key} numbers this equation"
        )
        assert_equation_says(key, value, require, forbid)
        checked.add(path[1:])
    for path, stated in SENTINEL_EVIDENCE.items():
        if path[0] != prefix:
            continue
        key = _source_of(path)
        assert table_value(path) == COULD_NOT_EXTRACT, (
            f"{list(path)} == {table_value(path)!r}; the source states this in prose / an "
            f"unnumbered display, so it must be {COULD_NOT_EXTRACT!r}"
        )
        assert re.search(stated(), source_prose(key), re.S), f"{key} no longer states {list(path)}"
        checked.add(path[1:])
    for path, value in _walk(root):
        if _is_eq_path(path) and value != COULD_NOT_EXTRACT:
            assert path in checked, f"{prefix}{list(path)} == {value!r} is not checked at all"


_PROSE_REF = re.compile(r"\bEqs?\.\s*((?:\(\d+[a-z]?\)(?:\s*(?:-|,|and)\s*)?)+)")
_APPROX = r"\\approx|\\simeq|\\mathcal\{O\}\(|\bO\("


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3}


def _overlap(desc: set[str], name: str) -> int:
    """How many words of a table eq key the description names (shared 4-letter stem)."""
    return sum(any(w[:4] == k[:4] for w in desc) for k in _words(name.replace("_", " ")))


def _cited_labels(run: str, recorded: dict[str, str]) -> list[str]:
    """``(25)-(29)`` -> both endpoints plus every recorded label between them."""
    labels = re.findall(r"\d+[a-z]?", run)
    if "-" in run and len(labels) == 2 and all(x.isdigit() for x in labels):
        lo, hi = map(int, labels)
        between = {v for v in recorded.values() if v.isdigit() and lo < int(v) < hi}
        labels = [labels[0], *sorted(between, key=int), labels[1]]
    return labels


#: M0-F10: where a source PRESENTS a first-order relation as one block, the ``linearized``
#: item naming it must list every equation of the block.  ``item words -> the source's own
#: introducing sentence``; the block runs from that sentence to the next section heading, and
#: its equations are the tags printed in it.  Delama: "The linearized error dynamics about
#: eps_k = 0 is then derived as follows" introduces (35) eps_{k+1} ~ A_hat eps + B_hat eta and
#: the two matrices it is written in, (36) A_hat and (37) B_hat; IV-E follows.
_LINEARIZED_PRESENTATION = {
    "delama": {"error dynamics": r"The linearized error dynamics about"},
}
_SECTION_HEADING = re.compile(r" [IVX]+-[A-Z] [A-Z][a-z]")
_PRINTED_TAG = re.compile(r"\$ \((\d+[a-z]?)\)")


def check_linearized_blocks(key: str) -> None:
    row = CONVENTIONS[key]
    recorded = {n: v for n, v in row["eq"].items() if v != COULD_NOT_EXTRACT}
    prose = source_prose(key) if key != "gtsam" else ""
    for words, intro in _LINEARIZED_PRESENTATION.get(key, {}).items():
        at = re.search(intro, prose)
        assert at, f"{key} no longer introduces its {words} block with {intro!r}"
        end = _SECTION_HEADING.search(prose, at.end())
        block = _PRINTED_TAG.findall(prose[at.end():end.start() if end else None])
        assert block, (key, words)
        items = [x for x in row["linearized"] if x.startswith(words)]
        assert len(items) == 1, (key, words, row["linearized"])
        cited = [lb for run in _PROSE_REF.findall(items[0]) for lb in _cited_labels(run, recorded)]
        assert sorted(cited) == sorted(block), (
            f"{key} linearized item {items[0]!r} cites {cited}; the source presents {block}")


def check_prose_equation_refs(key: str) -> None:
    """Every "Eq. (n)" inside an ``exact`` / ``linearized`` / ``relaxation`` item must cite an
    equation the row records under a key the item's own words name -- and no uncited key may
    be named better (M0-F06: these refs used to be checked only for existing).  The cited
    equation's own LaTeX decides the class: an ``exact`` item may cite no approximation."""
    row = CONVENTIONS[key]
    recorded = {n: v for n, v in row["eq"].items() if v != COULD_NOT_EXTRACT}
    prose = source_prose(key) if key != "gtsam" else ""
    fields = [("exact", x) for x in row["exact"]] + [("linearized", x) for x in row["linearized"]]
    if row.get("relaxation"):
        fields.append(("exact", row["relaxation"]))
    for kind, item in fields:
        runs = _PROSE_REF.findall(item)
        assert runs, f"{key} {kind} item {item!r} cites no equation"
        desc = _words(item[:item.index("Eq")])
        cited: dict[str, set[str]] = {}
        for run in runs:
            for label in _cited_labels(run, recorded):
                names = {n for n, v in recorded.items()
                         if v == label or re.fullmatch(re.escape(label) + r"[a-z]", v)}
                assert names, f"{key} {kind} item {item!r} cites Eq. ({label}), which the row does not record"
                cited[label] = names
        uncited = [n for n in recorded if not any(n in ns for ns in cited.values())]
        rival = max((_overlap(desc, n) for n in uncited), default=0)
        for label, names in cited.items():
            best = max(_overlap(desc, n) for n in names)
            assert best >= 1 and best >= rival, (
                f"{key} {kind} item {item!r}: Eq. ({label}) is recorded as {sorted(names)}, "
                f"which the item does not name (overlap {best}, an uncited key scores {rival})"
            )
        latex = {label: equation_latex(key, label) for label in cited}
        if kind == "exact":
            for label, lx in latex.items():
                assert not re.search(_APPROX, lx), f"{key} exact item cites Eq. ({label}), an approximation"
        else:
            def first_order(label, lx):
                at = prose.find(f"$ ({label})")
                before = prose[max(0, at - 400):at] if at != -1 else ""
                return bool(re.search(_APPROX, lx) or re.search(r"approximat|first.order", before)
                            or re.search(r"\\delta\\mathbf\{b\}", lx))
            assert any(first_order(lb, lx) for lb, lx in latex.items()), (
                f"{key} linearized item {item!r} cites no first-order relation")


# -- citations --------------------------------------------------------------------------

_ARXIV_CITATION = re.compile(
    r"arXiv:(?P<id>\d{4}\.\d{4,5}v\d+) \[(?P<cat>[\w.-]+)\], (?P<date>\d{1,2} \w{3} \d{4})"
    r"(?:; (?P<venue>.+?))? \((?P<authors>[^()]+?) -- (?P<title>[^()]+)\)")
_GTSAM_CITATION = re.compile(
    r"GTSAM docs, '(?P<title>[^']+)', (?P<url>\S+); behaviour re-verified against the "
    r"installed gtsam (?P<version>\S+) wheel")
CROSSREF = {"10.1177/02783649241269337": "crossref_10.1177_02783649241269337.json"}


def _ascii(text: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _squash(text: str) -> str:
    return re.sub(r"\W+", "", _ascii(_html.unescape(text))).lower()


def check_citation(key: str) -> None:
    """Every part of the citation string is decided by a cached source (M0-F06/F07): arXiv id,
    category, date, authors (order included) and title by the archived arXiv render; a journal
    fragment by the cached Crossref record or the arXiv render's own acceptance note; the GTSAM
    doc's title by the page itself and the version by the installed package."""
    row = CONVENTIONS[key]
    assert row["source_url"] == SOURCE_FIXTURES[key][1]
    raw = source_raw(key)
    if key == "gtsam":
        m = _GTSAM_CITATION.fullmatch(row["citation"])
        assert m, row["citation"]
        page = re.search(r"<title>([^<]*?) - GTSAM Docs</title>", raw)
        assert page and m.group("title") == _html.unescape(page.group(1)), (m.group("title"), page)
        assert "https://" + m.group("url") == row["source_url"], m.group("url")
        assert m.group("version") == _installed_gtsam_version() == row["package_version"]
        return
    m = _ARXIV_CITATION.fullmatch(row["citation"])
    assert m, f"CONVENTIONS[{key!r}]['citation'] is not 'arXiv:<id> [<cat>], <date>[; venue] (<authors> -- <title>)'"
    stamp = f"arXiv:{m.group('id')} [{m.group('cat')}] {m.group('date')}"
    assert stamp in raw, f"{key}: the archived render does not print {stamp!r}"
    assert SOURCE_FIXTURES[key][1].endswith(m.group("id")), (m.group("id"), SOURCE_FIXTURES[key][1])
    printed = [re.sub(r"\\affilnum.*|\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", x))).strip()
               for x in re.findall(r'class="ltx_personname">(.*?)</span>', raw, re.S)]
    names = [n for x in printed for n in re.split(r",\s*|\s+and\s+", x) if n.strip()]
    surnames = [_ascii(n.split()[-1]) for n in names]
    assert [x.strip() for x in m.group("authors").split(",")] == surnames, (m.group("authors"), surnames)
    title = re.search(r"<title>(.*?)</title>", raw, re.S)
    assert title and _squash(m.group("title")) == _squash(title.group(1)), (m.group("title"), title)
    venue = m.group("venue")
    if venue is None:
        return
    doi = re.fullmatch(r"(?P<head>.+), DOI (?P<doi>10\.\d{4,}/\S+)", venue)
    assert doi, venue
    if doi.group("doi") in CROSSREF:
        record = json.loads((_FIXTURES / CROSSREF[doi.group("doi")]).read_text())["message"]
        assert record["DOI"].lower() == doi.group("doi").lower()
        assert _squash(record["title"][0]) == _squash(m.group("title")), record["title"]
        assert [_ascii(a["family"]) for a in record["author"]] == surnames, record["author"]
        j = re.fullmatch(r"(?P<abbr>[A-Z]+) (?P<vol>\d+)\((?P<iss>\d+)\):(?P<pages>\d+-\d+), (?P<year>\d{4})",
                         doi.group("head"))
        assert j, doi.group("head")
        assert (j.group("vol"), j.group("iss"), j.group("pages")) == (
            record["volume"], record["issue"], record["page"]), (j.groups(), record["volume"])
        assert int(j.group("year")) == record["journal-issue"]["published-print"]["date-parts"][0][0]
        # the abbreviation is the source's own: its reference list writes "<journal> (IJRR)"
        journal = re.sub(r"^The ", "", record["container-title"][0])
        assert f"{journal} ({j.group('abbr')})" in source_prose(key), (journal, j.group("abbr"))
    else:
        note = f"for {doi.group('head')}, DOI:{doi.group('doi')}"
        assert note in source_prose(key), f"{key}: the archived render does not print {note!r}"


def _check_source_row(key: str) -> None:
    row = CONVENTIONS[key]
    _check_eq_entries(key, row)
    for path, value in _walk(row):
        if _is_eq_path(path) and value == COULD_NOT_EXTRACT:
            assert (key, *path) in SENTINEL_EVIDENCE, (key, path)
    # Any "Eq. (n)" written into a prose field must be a real equation of this source ...
    for path, value in _walk(row):
        if _is_eq_path(path):
            continue
        items = (value,) if isinstance(value, str) else (
            value if isinstance(value, tuple) and all(isinstance(x, str) for x in value) else ())
        for item in items:
            for run in _PROSE_REF.findall(item):
                for label in re.findall(r"\d+[a-z]?", run):
                    equation_latex(key, label)
    # ... and in the exact / linearized / relaxation items it must be the RIGHT one,
    # with no equation of a presented first-order block left out (M0-F10).
    check_prose_equation_refs(key)
    check_linearized_blocks(key)
    # Error handedness and bias-correction side, decided from the source for EVERY row
    # (M0-F06: rai's two were checked only against the enum).
    assert row["error_handedness"] == decide_error_handedness(key), (key, row["error_handedness"])
    assert row["bias_correction_side"] == decide_bias_correction_side(key), (
        key, row["bias_correction_side"])
    check_citation(key)


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
        _check_eq_entries(key, row)


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
        # None means "this source has no such object": the dependent fields follow it.
        assert (row["tangent_order"] is None) == (row["tangent_slots"] is None), key
        if row["element_order"] is None:
            assert row["ctor_order"] is None, key
        if row["bias_order"] is None:
            assert row["bias_dim"] is None, key
        else:
            # each physical IMU bias is a 3-vector; virtual channels have their slice widths
            virtual = row["bias_virtual_slices"]
            assert row["bias_dim"] == 3 * (len(row["bias_order"]) - len(virtual)) + sum(
                sl.stop - sl.start for sl in virtual), key
        names = set(row["bias_order"] or ())
        assert set(row["bias_zero_process_noise"]) <= names, key
        assert all(sl.stop <= (row["bias_dim"] or 0) for sl in row["bias_virtual_slices"]), key
        # a rank-1 criterion (and a tightness story) exists exactly for the certifiable,
        # exp/log-free formulations
        assert (row["rank1_criterion"] is None) == row["exp_log_essential"], key
        assert bool(row["tightness_degrades_with"]) == (row["rank1_criterion"] is not None), key

    for key, row in INCOMPATIBILITIES.items():
        for side in ("a", "b"):
            check(f"INCOMPATIBILITIES[{key!r}][{side!r}]", {"eq": row[side]["eq"]})
            assert row[side]["says"]
        assert isinstance(row["blocks"], tuple) and row["blocks"]
        assert row["where"] and row["consequence"]


# ======================================================================================
# The instruments themselves (M0-F07): fixture bytes, and a parser that invents nothing.
# ======================================================================================

@pytest.mark.claim("M0-F07")
def test_fixture_provenance():
    """Every cached source is the document its provenance record says, byte for byte.

    This also settles the rai size discrepancy of pass 1: 462 407 is the byte size, 456 935
    the number of characters after UTF-8 decoding (the page carries 2 616 non-ASCII
    characters), so both figures describe the same file."""
    import hashlib
    record = json.loads((_FIXTURES / "provenance.json").read_text())
    listed = {k for k in record if not k.startswith("_")}
    on_disk = {p.name for p in _FIXTURES.iterdir() if p.name != "provenance.json"}
    assert listed == on_disk, (listed ^ on_disk)
    for name in listed:
        blob = (_FIXTURES / name).read_bytes()
        raw = gzip.decompress(blob) if name.endswith(".gz") else blob
        assert len(raw) == record[name]["bytes"], name
        assert len(raw.decode("utf-8")) == record[name]["utf8_chars"], name
        assert hashlib.sha256(raw).hexdigest() == record[name]["sha256"], name
    for key, (name, url) in SOURCE_FIXTURES.items():
        assert record[name]["url"] == url, key
    for key, name in PDF_EXCERPTS.items():
        _, header = pdf_excerpt(key)
        assert f"# source_url:      {record[name]['url']}" in header
        assert SOURCE_FIXTURES[key][1].replace("/html/", "/pdf/") == record[name]["url"]
        assert record[name]["note"].split("PDF sha256 ")[1].split(",")[0] in header
    assert not list(_ROOT.glob("tests/**/*.pdf")), "a PDF was committed next to the tests"


@pytest.mark.claim("M0-F07")
def test_equation_parser_never_invents_a_label():
    """The arXiv index holds exactly the tags the HTML prints -- no positional sub-letters --
    and every label reached through a PDF excerpt is printed verbatim in that excerpt."""
    for key in ("rai", "delama", "barfoot"):
        printed = set(_LATEXML_TAG.findall(source_raw(key)))
        assert set(source_equations(key)) == printed, (key, set(source_equations(key)) ^ printed)
    for key, name in PDF_EXCERPTS.items():
        text = (_FIXTURES / name).read_text(encoding="utf-8")
        labels, _ = pdf_excerpt(key)
        assert labels and all(f"({lb})" in text for lb in labels), labels
        # where the HTML does print a label, the PDF puts the same LHS beside it: the two
        # renders number this block identically
        for lb, lhs in labels.items():
            if lb in source_equations(key):
                assert canon(source_equations(key)[lb]).split("=", 1)[0] == lhs, (lb, lhs)
    # The rows pass 2 found misfiled (N-1) now sit where the source prints them.
    r25a, r25b = equation_latex("rai", "25a"), equation_latex("rai", "25b")
    assert r"\mathbf{R}_{i}\mathbf{r}^{v}_{ij}" in r25a and r"\mathbf{r}^{p}" not in r25a
    assert r"\mathbf{R}_{i}\mathbf{r}^{p}_{ij}" in r25b and r"\mathbf{r}^{v}" not in r25b
    assert r"\Delta\bar{\mathbf{v}}" not in equation_latex("rai", "37b")
    assert r"\mathbf{z}^{p}_{i}" in equation_latex("rai", "37b")
    for key, invented in (("rai", "25c"), ("barfoot", "30b")):
        with pytest.raises(AssertionError):
            equation_latex(key, invented)
