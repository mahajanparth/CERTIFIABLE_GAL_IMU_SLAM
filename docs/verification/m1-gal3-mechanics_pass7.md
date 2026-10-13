# M1 seventh-pass adversarial verification (IN PROGRESS)

Branch `m1-gal3-mechanics`, HEAD 1cd8562. Independent references only (numpy, scipy, mpmath,
gtsam 4.3.0; no imports from `src` or `tests` in any reference). Results are being filled in.

## Early result

- **M1-C61 REFUTED.** H'(Dt1) H'(Dt2) = (I, g (Dt1 + Dt2), -0.5 g (Dt1^2 + Dt2^2), 0), while
  H'(Dt1 + Dt2) has position -0.5 g (Dt1 + Dt2)^2. They differ by exactly g Dt1 Dt2. Measured on
  2000 draws (random g, Dt in (-1, 1)): max error **16.0**, which equals the predicted max |g| |Dt1 Dt2|.
  For g = (0, 0, -9.81) and Dt1 = Dt2 = 0.5 the error is **2.4525**. The bound claimed is 1e-12.
