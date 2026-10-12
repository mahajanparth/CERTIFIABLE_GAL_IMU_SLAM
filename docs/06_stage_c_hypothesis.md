# Stage C — seeded hypothesis (DORMANT)

**Status: no execution work.** Stage C was scoped out. This branch exists so the pre-flight analysis
survives in version control and a later decision starts from a real document rather than a blank page.

## The obstruction, located precisely

Delama et al. define the equivariant error (Eq. 29) and its chart (Eq. 30/31/34):

```
e_k   = phi(Xhat_k^-1, xi_k) = ( Upsilon_k Chat_k^-1 ,  Ad_Chat_k b_k + gammahat_k )
eps_k = ( log(Upsilon_k Upsilonhat_k^-1) ,  -J_L(eps_Upsilon)^-1 Ad_Upsilon (b_k - bhat_k) )
```

Split the bias error into its two factors:

| factor | status | why |
| --- | --- | --- |
| `Ad_Upsilon (b - bhat)` | **polynomial** | verified — see below |
| `J_L(eps_Upsilon)^-1` | **the obstruction** | transcendental: `sin`/`cos` of `\|\|omega\|\|` via `Gamma1, Gamma2, kappa1..3` |

So the barrier is *not* the adjoint, which is where the project's original notes expected it.

## What is already verified (F8, reproducible via `experiments/preflight/gal3_checks.py`)

The Gal(3) adjoint in closed form, confirmed numerically to machine precision:

```
phi'   = R phi
u'     = R u + [v]x R phi
w'     = R w - t R u + [p - t v]x R phi + alpha v
alpha' = alpha                                   (time coordinate is Ad-invariant)
```

Raw degree is **3** (`v (x) R (x) phi`), so not directly QCQP. But a **two-stage lifting closes it**:
`a1 = R phi`, `a2 = [v]x a1`, `a3 = [p]x a1`, `a4 = R u` — each degree 2, after which every adjoint
equation is degree <= 2. These are exactly the `R db`, `[v]x R db`, `[p]x R db` products the original
Stage C notes asked to audit.

Also measured, over 20000 random `gal(3)` elements **with nonzero time coordinate**:
- Cayley is **total** on `gal(3) -> Gal(3)` — zero failures.
- Its only denominator is `det(I5 - A/2) = 1 + ||phi||^2/4 > 0`, max deviation `9.1e-13`.

That is the exact analogue of Rai Eq. (17), and Rai's Cayley right Jacobian
`J_Cay(phi) = (I - phi^/2)/(1 + ||phi||^2/4)` is likewise **rational**.

Delama Sec. V-A adds that when `Dt_hat = Dt` the time-error coordinate is **zero**, so the zero-time
9-dim Cayley block (pre-flight F3) is the relevant chart.

## The hypothesis — NOT TESTED, NOT CLAIMED

> Does `J_L^Cay(eps)^-1 Ad_Upsilon (b - bhat)` clear to a polynomial of total degree <= 2 under
> staged lifting, with a denominator that is positive on the whole feasible set?

If yes, Stage C is reachable: a Cayley-charted equivariant Galilean factor that is QCQP-representable.
If no, the minimal counterexample is itself the result — an explicit obstruction showing equivariant
bias coupling cannot survive finite polynomial lifting.

**Either answer is publishable. Neither is established.** Nothing here may be cited as a result.

## If Stage C is resumed, start here

1. Derive `J_L^Cay` for Gal(3) — the Cayley analogue of Delama Eq. (10). Determine whether it is
   rational and what its denominator is.
2. Degree-audit `J_L^Cay(eps)^-1 Ad_Upsilon (b - bhat)` in scalar coordinates.
3. Prove lifting equivalence **in both directions** — not merely that ground truth satisfies the
   constraints, but that no spurious feasible points are admitted.
4. Only then consider an SDP. Note the 20-dim symmetry group makes this far larger than Stage B.

## Licence note

`aau-cns/equivariant-preintegration` is **not OSI-licensed**: BSD-2 plus a no-commercial-use clause,
mandatory citation, and no patent grant (`SPDX: NOASSERTION`). Its dependency **`Lie++` is
Apache-2.0** and is where all the Gal(3) math actually lives — that is the safe thing to read or port.
