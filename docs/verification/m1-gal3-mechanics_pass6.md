# M1 sixth-pass adversarial verification (Gal(3) mechanics and IMU dynamics)

Branch `m1-gal3-mechanics`, verified at `9cda195`. gtsam 4.3.0, numpy, scipy, sympy, mpmath.

Status: IN PROGRESS.

- Clean suite at the start of the pass: **120 passed** (197 s).
- M1-C60 (brief only): REFUTED. sin(theta)/theta minus 1 - theta^2/6 + theta^4/120 has relative
  error 3.10e-12 at theta = 0.05 (mpmath, 50 digits), 3.1e4 times the claimed 1e-16; the bound
  already fails above theta = 8.9e-3.
