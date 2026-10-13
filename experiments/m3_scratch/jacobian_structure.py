"""M3 scratch: structural zeros/equalities of GTSAM's Galilean right-chart bias Jacobian J (10x6,
project order: rows (omega, nu=vel, rho=pos, alpha), columns [b_g; b_a]), by finite differences.

  J[0:3, 3:6] == 0   rotation independent of accel bias
  J[9, :]     == 0   time coordinate independent of bias
  J[3:6, 3:6] == J[0:3, 0:3]   (vel/accel block equals rot/gyro block)

The third follows from GTSAM doc Eq.(35) J_{k+1} = Ad_{V_k^-1} J_k - C_k with Ad from Eq.(9) and
C_k = J_R(x_k) h E (Eq.(20) E): with J^{omega,a} = 0 the (nu,a) block obeys the same recursion as
the (omega,g) block, provided the (nu,nu) and (omega,omega) blocks of the Gal(3) right Jacobian
coincide -- that last step is checked here numerically, not proven.

Run:  python -m experiments.m3_scratch.jacobian_structure
"""
import numpy as np
import gtsam
from .instance import params, _gt_bias, gal, jacobian, build

rng = np.random.default_rng(3)


def main():
    inst = build(np.zeros(6))
    for e, E in enumerate(inst["edges"]):
        J = E.J
        print(f"fixture edge {e}: |J[0:3,3:6]|={np.abs(J[0:3, 3:6]).max():.1e} |J[9]|={np.abs(J[9]).max():.1e} "
              f"|J_va - J_R|={np.abs(J[3:6, 3:6] - J[0:3, 0:3]).max():.1e} (|J_R|={np.abs(J[0:3, 0:3]).max():.3f})")
    for trial in range(5):
        bbar = rng.normal(size=6) * 0.05
        pim = gtsam.PreintegratedImuMeasurementsG(params(), _gt_bias(bbar))
        for k in range(150):
            pim.integrateMeasurement(rng.normal(size=3) * 3 + np.array([0, 0, 9.8]), rng.normal(size=3) * 1.5, 2.5e-3)
        U = gal(pim.deltaRij().matrix(), np.array(pim.deltaVij()), np.array(pim.deltaPij()), pim.deltaTij())
        J = jacobian(pim, bbar, U)
        print(f"time-varying stream {trial}: |J[0:3,3:6]|={np.abs(J[0:3, 3:6]).max():.1e} |J[9]|={np.abs(J[9]).max():.1e} "
              f"|J_va - J_R|={np.abs(J[3:6, 3:6] - J[0:3, 0:3]).max():.1e} (|J_R|={np.abs(J[0:3, 0:3]).max():.3f})")


if __name__ == "__main__":
    main()
