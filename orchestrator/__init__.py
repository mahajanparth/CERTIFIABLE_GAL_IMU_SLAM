"""Progress ledger, gates and agent dispatch for the Stage B research programme.

Design rule: progress is COMPUTED FROM EVIDENCE, never self-reported. An agent asserting
"M3 is 80% done" carries no weight here. Status is derived from facts this package checks
itself -- whether a claim is bound to a test, whether that test passes when actually run,
whether an independent verifier returned a verdict, and whether that verifier caught the
defect deliberately planted in its brief.
"""
__all__ = ["ledger", "evidence", "gates", "dispatch", "render"]
