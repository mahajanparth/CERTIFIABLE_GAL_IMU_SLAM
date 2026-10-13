"""M3: the QCQP of the direct-product Galilean IMU preintegration factor.

Modules
-------
fixture         the M2_TRAJ_V1 trajectory constants and closed-form ground truth
gtsam_boundary  the single place GTSAM conventions are converted (bias order, Gal3 ctor,
                NavState vs Gal3 tangent order)
instance        the 3-KF / 2-edge GNSS-anchored instance (bbar = b_true or 0)
lifting         index map, z(state), state(z)
forms           linear / quadratic form tensors over z
cost            residual forms, terms {F_t, W_t}, Q
constraints     every constraint family {A_k, b_k}
"""
from .instance import build_instance  # noqa: F401
from .lifting import BASE, ANISO, Layout, State, z_of_state, state_of_z, truth_state  # noqa: F401
