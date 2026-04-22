import numpy as np

from code.orbital_state import OrbitalState


def _is_angle_close(angle: float, target: float, tol: float = 1e-6) -> bool:
    d = (angle - target + np.pi) % (2.0 * np.pi) - np.pi
    return abs(d) < tol


def compute_hohmann_transfer(
    orbit_initial: OrbitalState,
    orbit_final: OrbitalState,
    angle_tol: float = 1e-6,
    plane_tol: float = 1e-8,
):
    """
    Compute a Hohmann transfer between two coplanar, coaxial elliptical/circular orbits.

    Returns
    -------
    transfer_orbit : OrbitalState
    delta_v1 : float
    delta_v2 : float
    delta_v_total : float
    tof : float
    """

    # --- Ensure elements ---
    orbit_initial.ensure_elements()
    orbit_final.ensure_elements()

    mu = orbit_initial.mu

    # --- Basic checks ---
    if orbit_initial.mu != orbit_final.mu:
        raise ValueError("Initial and final orbits must use the same mu.")

    if orbit_initial.e >= 1.0 or orbit_final.e >= 1.0:
        raise ValueError("Only elliptical/circular orbits (e < 1) are supported.")

    # --- Coplanarity / alignment ---
    if abs(orbit_initial.i - orbit_final.i) > plane_tol:
        raise ValueError("Inclination mismatch.")

    if abs(orbit_initial.Omega - orbit_final.Omega) > plane_tol:
        raise ValueError("RAAN mismatch.")

    if abs(orbit_initial.omega - orbit_final.omega) > plane_tol:
        raise ValueError("Argument of periapsis mismatch.")

    # --- Check apsis locations ---
    theta_i = orbit_initial.theta
    theta_f = orbit_final.theta

    if theta_i is None or theta_f is None:
        raise ValueError("Both orbits must define theta.")

    initial_at_peri = _is_angle_close(theta_i, 0.0, angle_tol)
    initial_at_apo = _is_angle_close(theta_i, np.pi, angle_tol)

    if not (initial_at_peri or initial_at_apo):
        raise ValueError("Initial orbit must be at periapsis or apoapsis.")

    final_at_peri = _is_angle_close(theta_f, 0.0, angle_tol)
    final_at_apo = _is_angle_close(theta_f, np.pi, angle_tol)

    if not (final_at_peri or final_at_apo):
        raise ValueError("Final orbit must be at periapsis or apoapsis.")

    # --- Radii using properties ---
    r_i = orbit_initial.r_p if initial_at_peri else orbit_initial.r_a
    r_f = orbit_final.r_p if final_at_peri else orbit_final.r_a

    if r_i is None or r_f is None:
        raise ValueError("Could not determine apsis radii.")

    # --- Transfer geometry ---
    r_p = min(r_i, r_f)
    r_a = max(r_i, r_f)

    a_t = 0.5 * (r_p + r_a)
    e_t = (r_a - r_p) / (r_a + r_p)

    # Determine starting point on transfer orbit
    if abs(r_i - r_p) < 1e-10:
        theta_t = 0.0  # start at periapsis
    else:
        theta_t = np.pi  # start at apoapsis

    transfer = OrbitalState(
        mu=mu,
        a=a_t,
        e=e_t,
        i=orbit_initial.i,
        Omega=orbit_initial.Omega,
        omega=orbit_initial.omega,
        theta=theta_t,
    )

    transfer.compute_period(update=True)
    transfer.compute_state_vectors(update=True)

    # --- Velocity calculations ---
    # Vis-viva: v = sqrt(mu*(2/r - 1/a))

    # Initial orbit velocity at burn
    v_i = np.sqrt(mu * (2.0 / r_i - 1.0 / orbit_initial.a))

    # Transfer orbit velocity at start
    v_t1 = np.sqrt(mu * (2.0 / r_i - 1.0 / a_t))

    # Final orbit velocity at arrival
    v_f = np.sqrt(mu * (2.0 / r_f - 1.0 / orbit_final.a))

    # Transfer orbit velocity at arrival
    v_t2 = np.sqrt(mu * (2.0 / r_f - 1.0 / a_t))

    delta_v1 = v_t1 - v_i
    delta_v2 = v_f - v_t2
    # delta_v_total = delta_v1 + delta_v2

    # --- Time of flight (half period) ---
    tof = np.pi * np.sqrt(a_t**3 / mu)

    return transfer, delta_v1, delta_v2, tof
