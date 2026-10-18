"""Thread caps for the test process and every subprocess it spawns.

OpenBLAS oversubscription on this loaded machine makes runs crawl or hang: each pytest
process (and each mutant copy run by ``tests/test_mutation.py``, which inherits this
environment) would otherwise start one BLAS thread per core.  The caps must be set before
numpy is first imported, which is why they live here and not in a fixture.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
