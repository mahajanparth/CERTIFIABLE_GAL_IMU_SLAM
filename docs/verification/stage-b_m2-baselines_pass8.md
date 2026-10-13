# Stage-B verification, pass 8: M2 baselines (`stage-b/m2-baselines` @ `5de592a`)

Eighth independent adversarial pass. **In progress**: this skeleton is committed early and
extended as the pass proceeds.

Scope: rulings on M2-F16..F20; every claim in `docs/claims/M2.yaml` (C01-C25); one claim that
exists only in the pass brief, **M2-C31**; a mutation sweep of new file-level mutants; a replay of
earlier mutants as real file edits; tests that cannot fail.

Rules kept: no derivation narrative (`docs/00_*`, `docs/02_*`), no `seeds*.yaml`, nothing under
`.cgis_SEEDS_BACKUP*` was read. All scratch code and mutant copies live in the session scratchpad.
No `src/`, `tests/` or `docs/claims/` file is edited.

Clean tree at `5de592a`: full suite **203 passed** in 53.3 s.
