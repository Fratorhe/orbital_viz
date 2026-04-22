from dataclasses import asdict, dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from code.constants import deg2rad, mu_sun, to_days

Vector3 = npt.NDArray[np.float64]


def rotx(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ]
    )


def rotz(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def print_type_orbit(eccentricity: float) -> None:
    if eccentricity < 1.0:
        print("Elliptical Orbit")
    else:
        print("Hyperbolic Orbit")


@dataclass
class OrbitalState:
    # Central body parameter
    mu: float

    # Classical orbital elements
    a: Optional[float] = None
    e: Optional[float] = None
    i: Optional[float] = None
    Omega: Optional[float] = None
    omega: Optional[float] = None
    theta: Optional[float] = None
    h: Optional[float] = None

    # State vectors
    r_vec: Optional[Vector3] = field(default=None)
    v_vec: Optional[Vector3] = field(default=None)
    h_vec: Optional[Vector3] = field(default=None)
    e_vec: Optional[Vector3] = field(default=None)

    def __post_init__(self):
        for key in ["r_vec", "v_vec", "h_vec", "e_vec"]:
            value = getattr(self, key)
            if value is not None:
                value = np.asarray(value, dtype=float)
                if value.shape != (3,):
                    raise ValueError(f"{key} must be a 3-element vector")
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def to_dict(self):
        return asdict(self)

    @property
    def period_in_days(self):
        if self.period is None:
            return None
        return self.period * to_days

    def update(self, other: Union["OrbitalState", dict], overwrite: bool = True):
        if isinstance(other, OrbitalState):
            data = other.to_dict()
        elif isinstance(other, dict):
            data = other
        else:
            raise TypeError("update expects OrbitalState or dict")

        for key, value in data.items():
            if not hasattr(self, key):
                continue
            if value is None:
                continue

            current_value = getattr(self, key)
            if not overwrite and current_value is not None:
                continue

            if key in ["r_vec", "v_vec", "h_vec", "e_vec"]:
                value = np.asarray(value, dtype=float)
                if value.shape != (3,):
                    raise ValueError(f"{key} must be a 3-element vector")

            setattr(self, key, value)

        return self

    @property
    def period(self) -> Optional[float]:
        if self.a is None:
            return None

        period = 2.0 * np.pi * self.a ** (3.0 / 2.0) / np.sqrt(self.mu)

        return period

    def compute_elements(self, update: bool = True, verbose: bool = False):
        if self.r_vec is None or self.v_vec is None:
            raise ValueError("r_vec and v_vec are required to compute orbital elements")

        r_vec = np.asarray(self.r_vec, dtype=float)
        v_vec = np.asarray(self.v_vec, dtype=float)

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        if r == 0.0:
            raise ValueError("r_vec has zero norm")

        v_r = np.dot(v_vec, r_vec) / r

        h_vec = np.cross(r_vec, v_vec)
        h = float(np.linalg.norm(h_vec))

        if h == 0.0:
            raise ValueError("Angular momentum is zero; orbit is undefined")

        i = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

        N_vec = np.cross([0.0, 0.0, 1.0], h_vec)
        N = np.linalg.norm(N_vec)

        # RAAN
        if N > 1e-12:
            Omega = np.arccos(np.clip(N_vec[0] / N, -1.0, 1.0))
            if N_vec[1] < 0:
                Omega = 2.0 * np.pi - Omega
        else:
            Omega = 0.0

        # Eccentricity vector
        e_vec = ((v**2 - self.mu / r) * r_vec - r * v_r * v_vec) / self.mu
        e = float(np.linalg.norm(e_vec))

        # Argument of periapsis
        if N > 1e-12 and e > 1e-12:
            omega = np.arccos(np.clip(np.dot(N_vec, e_vec) / (N * e), -1.0, 1.0))
            if e_vec[2] < 0:
                omega = 2.0 * np.pi - omega
        else:
            omega = 0.0

        # True anomaly
        if e > 1e-12:
            theta = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
            if v_r < 0:
                theta = 2.0 * np.pi - theta
        else:
            theta = 0.0

        # Semi-major axis
        a = 1.0 / (2.0 / r - v**2 / self.mu)

        period = None
        if a > 0:
            period = 2.0 * np.pi * a ** (3.0 / 2.0) / np.sqrt(self.mu)

        elements = OrbitalState(
            mu=self.mu,
            a=a,
            e=e,
            i=i,
            Omega=Omega,
            omega=omega,
            theta=theta,
            h=h,
            h_vec=h_vec,
            e_vec=e_vec,
        )

        if verbose:
            print(f"Angular momentum h is {h:.3e} km^2/s")
            print(f"Orbit inclination is {i:.3f} rad")
            print(f"Orbit inclination is {i / deg2rad:.3f} degrees")

            if i > np.pi / 2:
                print("Since i greater than 90 degrees, this is a retrograde orbit.")
                print("Rotation of object is contrary to rotation of the body")
            else:
                print("Since i smaller than 90 degrees, this is a normal orbit.")
                print("Rotation of object is same to rotation of the body.")

            print(N_vec)
            print(f"{N:.3e}")
            print(f"Omega is {Omega:.3f} rad")
            print(f"Omega is {Omega / deg2rad:.3f} degrees")
            print(e_vec)
            print(f"The eccentricity of the orbit is: {e:.6f}")
            print_type_orbit(e)
            print(f"The periapsis argument is {omega:.3f} rad")
            print(f"The periapsis argument is {omega / deg2rad:.3f} degrees")
            print(f"The true anomaly is {theta:.3f} rad")
            print(f"The true anomaly is {theta / deg2rad:.3f} degrees")

        if update:
            self.update(elements)
            return self

        return elements

    def compute_state_vectors(self, update: bool = True):
        required = ["a", "e", "i", "Omega", "omega", "theta"]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(
                f"Missing orbital elements needed for state vector: {missing}"
            )

        a = float(self.a)
        e = float(self.e)
        i = float(self.i)
        Omega = float(self.Omega)
        omega = float(self.omega)
        theta = float(self.theta)

        h = np.sqrt(self.mu * a * (1.0 - e**2))

        r = h**2 / self.mu / (1.0 + e * np.cos(theta))

        # Perifocal coordinates
        r_pf = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
        v_pf = np.array(
            [
                -self.mu / h * np.sin(theta),
                self.mu / h * (e + np.cos(theta)),
                0.0,
            ]
        )

        R_Omega = rotz(-Omega)
        R_incl = rotx(-i)
        R_omega = rotz(-omega)

        Q = R_Omega @ R_incl @ R_omega

        r_vec = Q @ r_pf
        v_vec = Q @ v_pf
        h_vec = np.cross(r_vec, v_vec)

        state = OrbitalState(
            mu=self.mu,
            r_vec=r_vec,
            v_vec=v_vec,
            h=float(np.linalg.norm(h_vec)),
            h_vec=h_vec,
        )

        if update:
            self.update(state)
            return self

        return state

    def get_apses_line_points(self):
        """
        Return the endpoints of the line of apses in inertial coordinates.

        Returns
        -------
        (p1, p2) : tuple[np.ndarray, np.ndarray] or None
            Endpoints of the apses line.
            For elliptical orbits:
                p1 = apoapsis point
                p2 = periapsis point
            Returns None if the eccentricity vector is undefined.
        """
        self.ensure_elements()

        if self.e is None or self.a is None or self.e_vec is None:
            return None

        e_vec = np.asarray(self.e_vec, dtype=float)
        e_norm = np.linalg.norm(e_vec)

        if e_norm < 1e-10:
            return None

        e_hat = e_vec / e_norm

        if self.e < 1:
            r_p = self.a * (1 - self.e)
            r_a = self.a * (1 + self.e)

            p1 = -r_a * e_hat
            p2 = r_p * e_hat
        else:
            r_p = self.a * (1 - self.e)
            p1 = -2 * abs(self.a) * e_hat
            p2 = r_p * e_hat

        return p1, p2

    def ensure_elements(self):
        required = ["a", "e", "i", "Omega", "omega", "theta"]
        if any(getattr(self, name) is None for name in required):
            self.compute_elements(update=True, verbose=False)
        return self

    def ensure_state_vectors(self):
        if self.r_vec is None or self.v_vec is None:
            self.compute_state_vectors(update=True)
        return self

    @property
    def r(self) -> Optional[float]:
        """Instantaneous radius (from state vector if available)"""
        if self.r_vec is not None:
            return float(np.linalg.norm(self.r_vec))
        if self.a is not None and self.e is not None and self.theta is not None:
            return self.a * (1 - self.e**2) / (1 + self.e * np.cos(self.theta))
        return None

    @property
    def r_p(self) -> Optional[float]:
        """Periapsis radius"""
        if self.a is None or self.e is None:
            return None
        return self.a * (1.0 - self.e)

    @property
    def r_a(self) -> Optional[float]:
        """Apoapsis radius"""
        if self.a is None or self.e is None:
            return None
        if self.e >= 1.0:
            return None  # no apoapsis for hyperbolic orbits
        return self.a * (1.0 + self.e)

    @property
    def v_r(self) -> float:
        """
        Radial velocity component (km/s)
        """
        self.ensure_state_vectors()

        r_vec = self.r_vec
        v_vec = self.v_vec

        r = np.linalg.norm(r_vec)
        if r < 1e-12:
            raise ValueError("Radius is zero.")

        return float(np.dot(v_vec, r_vec) / r)

    @property
    def v_t(self) -> float:
        """
        Transverse (tangential) velocity component (km/s)
        """
        self.ensure_state_vectors()

        r = np.linalg.norm(self.r_vec)
        if r < 1e-12:
            raise ValueError("Radius is zero.")

        if self.h is None:
            # compute angular momentum if needed
            h_vec = np.cross(self.r_vec, self.v_vec)
            h = np.linalg.norm(h_vec)
        else:
            h = self.h

        return float(h / r)

    @property
    def flight_path_angle(self) -> float:
        """
        Flight path angle (rad)
        Positive when moving away from periapsis (outbound)
        """
        v_r = self.v_r
        v_t = self.v_t

        return float(np.arctan2(v_r, v_t))

    @property
    def r_hat(self) -> Vector3:
        self.ensure_state_vectors()
        r_vec = np.asarray(self.r_vec, dtype=float)
        r = np.linalg.norm(r_vec)
        if r < 1e-12:
            raise ValueError("Radius is zero.")
        return r_vec / r

    @property
    def h_hat(self) -> Vector3:
        self.ensure_state_vectors()

        if self.h_vec is not None:
            h_vec = np.asarray(self.h_vec, dtype=float)
        else:
            h_vec = np.cross(self.r_vec, self.v_vec)

        h = np.linalg.norm(h_vec)
        if h < 1e-12:
            raise ValueError("Angular momentum is zero.")
        return h_vec / h

    @property
    def t_hat(self) -> Vector3:
        return np.cross(self.h_hat, self.r_hat)

    @property
    def v_r_vec(self) -> Vector3:
        return self.v_r * self.r_hat

    @property
    def v_t_vec(self) -> Vector3:
        return self.v_t * self.t_hat

    def copy(self):
        return OrbitalState(**self.to_dict())


def compute_period(orbiting_object):
    a = orbiting_object.a
    return 2 * np.pi / np.sqrt(orbiting_object.mu) * a ** (3 / 2)


if __name__ == "__main__":
    earth = OrbitalState(mu=mu_sun, a=149597871, e=0)
    earth_period = compute_period(earth)
    earth_period_days = earth_period * to_days
    print(f"Earth takes: {earth_period_days} days")
