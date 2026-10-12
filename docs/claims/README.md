# Claim files

One YAML file per milestone, written by that milestone's **deriver**. Claims live here rather
than in `orchestrator/state.yaml` so that several milestones can be derived concurrently on
separate branches without three agents fighting over one file.

```yaml
milestone: M1
claims:
  - id: M1-C01
    statement: "A one-sentence MACHINE-CHECKABLE predicate -- something a test can decide."
    source: "Delama et al. Eq. (5)"      # or "DERIVED"
    kind: VERIFIED-PUBLISHED             # VERIFIED-PUBLISHED | DERIVED | ASSUMPTION | OPEN-HYPOTHESIS
    test_hint: "Inputs, expected result, tolerance."
    domain: "Any restriction: chart singularities, nonzero denominators, linearization validity."
```

Rules:
- A claim must be **decidable by a test**. "The formulation is sound" is not a claim;
  "`X @ gal_inv(X)` equals `I5` to `1e-12` for 1e4 random elements" is.
- Ids are `M<k>-C<nn>` and must match the milestone.
- Never guess an equation number. Write `could not extract` instead.
- Tag every claim with `kind` so published fact is never blurred with our own derivation.

The orchestrator reads these files, binds each id to a `@pytest.mark.claim` test, runs pytest,
and computes status. An unbound claim reads `UNTESTED` and blocks the milestone gate, so a claim
that gets skipped stays visible instead of quietly vanishing.
