"""
A two-wave hydrodynamic modulation model.
"""

import numpy as np
from rich.progress import track
import xarray as xr


def angular_frequency(g: float, k: float, u: float = 0) -> float:
    return np.sqrt(g * k) + k * u


def elevation(
    x: float, t: float, a: float, k: float, omega: float, wave_type: str = "linear"
) -> float:
    phase = k * x - omega * t
    if wave_type == "linear":
        return a * np.cos(phase)
    elif wave_type == "stokes":
        term1 = np.cos(phase)
        term2 = 0.5 * a * k * np.cos(2 * phase)
        term3 = (a * k) ** 2 * (3 / 8 * np.cos(3 * phase) - 1 / 16 * np.cos(phase))
        return a * (term1 + term2 + term3)
    else:
        raise ValueError("wave_type must be either 'linear' or 'stokes'")


def surface_slope(
    x: float, t: float, a: float, k: float, omega: float, wave_type: str = "linear"
) -> float:
    phase = k * x - omega * t
    ak = a * k
    if wave_type == "linear":
        slope = -ak * np.sin(phase)
    elif wave_type == "stokes":
        term1 = -ak * np.sin(phase)
        term2 = -(ak**2) * np.sin(2 * phase)
        term3 = -(ak**3) * (9 / 8 * np.sin(3 * phase) - 1 / 16 * np.sin(phase))
        slope = term1 + term2 + term3
    else:
        raise ValueError("wave_type must be either 'linear' or 'stokes'")
    return slope


def gravity(
    x: float,
    t: float,
    a: float,
    k: float,
    omega: float,
    g0: float = 9.8,
    wave_type: str = "linear",
) -> float:
    """Gravitational acceleration at the surface of a long wave.
    Supports both linear and Stokes wave types.
    """
    phase = k * x - omega * t
    eta = elevation(x, t, a, k, omega, wave_type=wave_type)

    if wave_type == "linear":
        return g0 * (
            1 - a * k * np.exp(k * eta) * (np.cos(phase) - a * k * np.sin(phase) ** 2)
        )
    elif wave_type == "stokes":
        return g0 * (
            1
            - a
            * k
            * (
                np.cos(phase)
                - a
                * k
                * np.sin(phase)
                * (
                    (1 - 1 / 16 * (a * k) ** 2) * np.sin(phase)
                    + a * k * np.sin(2 * phase)
                    + 9 / 8 * (a * k) ** 2 * np.sin(3 * phase)
                )
            )
            * np.exp(k * eta)
        )
    else:
        raise ValueError("wave_type must be either 'linear' or 'stokes'")


def orbital_horizontal_velocity(
    x: float, z: float, t: float, a: float, k: float, omega: float
) -> float:
    """Horizontal orbital velocity at depth z."""
    return a * omega * np.cos(k * x - omega * t) * np.exp(k * z)


def orbital_vertical_velocity(
    x: float, z: float, t: float, a: float, k: float, omega: float
) -> float:
    """Horizontal orbital velocity at depth z."""
    return a * omega * np.sin(k * x - omega * t) * np.exp(k * z)


def orbital_horizontal_acceleration(
    x: float, t: float, a: float, k: float, omega: float, wave_type: str = "linear"
) -> float:
    phase = k * x - omega * t
    if wave_type == "linear":
        eta = elevation(x, t, a, k, omega, wave_type)
        dU_dt = (
            a * omega**2 * np.exp(k * eta) * np.sin(phase) * (a * k * np.cos(phase) + 1)
        )
    elif wave_type == "stokes":
        eta = elevation(x, t, a, k, omega, wave_type)
        term1 = a * omega**2 * np.exp(k * eta) * np.sin(phase)
        term2 = (
            a**2
            * omega**2
            * k
            * np.exp(k * eta)
            * np.cos(phase)
            * (
                np.sin(phase)
                + a * k * np.sin(2 * phase)
                + (a * k) ** 2
                * (9 / 8.0 * np.sin(3 * phase) - 1 / 16.0 * np.sin(phase))
            )
        )
        dU_dt = term1 + term2
    else:
        raise ValueError("wave_type must be either 'linear' or 'stokes'")
    return dU_dt


def orbital_vertical_acceleration(
    x: float, t: float, a: float, k: float, omega: float, wave_type: str = "linear"
) -> float:
    phase = k * x - omega * t
    if wave_type == "linear":
        eta = elevation(x, t, a, k, omega, wave_type)
        dW_dt = (
            a
            * omega**2
            * np.exp(k * eta)
            * (a * k * np.sin(phase) ** 2 - np.cos(phase))
        )
    elif wave_type == "stokes":
        eta = elevation(x, t, a, k, omega, wave_type)
        term1 = -a * omega**2 * np.exp(k * eta) * np.cos(phase)
        term2 = (
            a**2
            * omega**2
            * k
            * np.exp(k * eta)
            * np.sin(phase)
            * (
                np.sin(phase)
                + a * k * np.sin(2 * phase)
                + (a * k) ** 2
                * (9 / 8.0 * np.sin(3 * phase) - 1 / 16.0 * np.sin(phase))
            )
        )
        dW_dt = term1 + term2
    else:
        raise ValueError("wave_type must be either 'linear' or 'stokes'")
    return dW_dt


def gravity_curvilinear(
    x: float,
    t: float,
    a: float,
    k: float,
    omega: float,
    g0: float = 9.8,
    wave_type: str = "linear",
) -> float:
    dU_dt = orbital_horizontal_acceleration(x, t, a, k, omega, wave_type)
    dW_dt = orbital_vertical_acceleration(x, t, a, k, omega, wave_type)
    slope = surface_slope(x, t, a, k, omega, wave_type)
    g = g0 * np.cos(slope) + dW_dt * np.cos(slope) + dU_dt * np.sin(slope)
    return g


def diff(x: np.ndarray) -> np.ndarray:
    """2nd order, centered difference"""
    dx = np.zeros_like(x)
    dx[..., 1:-1] = 0.5 * (x[..., 2:] - x[..., :-2])
    dx[..., 0] = 0.5 * (x[..., 1] - x[..., -1])
    dx[..., -1] = 0.5 * (x[..., 0] - x[..., -2])
    return dx


def rk4(f: callable, x0: np.ndarray, t0: float, dt: float) -> np.ndarray:
    """Solve x' = f(x,t) with x(t[0]) = x0 using RK4 integration for one time step."""
    x = np.zeros_like(x0)
    k1 = dt * f(x0, t0)
    k2 = dt * f(x0 + k1 / 2, t0 + dt / 2)
    k3 = dt * f(x0 + k2 / 2, t0 + dt / 2)
    k4 = dt * f(x0 + k3, t0 + dt)
    x = x0 + (k1 + 2 * (k2 + k3) + k4) / 6
    return x


class WaveModulationModel:
    """This class defines the two-wave hydrodynamic modulation model.

    Instantiate:

    >>> m = WaveModulationModel()

    Run the model:

    >>> m.run()

    Run the model and save the tendencies at each time step:

    >>> m.run(save_tendencies=True)

    Run the model with a linear elevation ramp:

    >>> m.run(ramp_type="linear")

    Run the model with a group elevation ramp:

    >>> m.run(ramp_type="groups")

    Get the model output as an Xarray dataset:

    >>> ds = m.to_xarray()
    """

    def __init__(
        self,
        a_long: float = 0.1,
        k_long: float = 1,
        grav0: float = 9.8,
        a_short: float | np.ndarray = 0.01,
        k_short: float | np.ndarray = 10,
        grid_size: int = 100,
        time_step: float = 1e-2,
        num_periods: int = 10,
        curvilinear: bool = True,
    ) -> None:
        """Initialize the wave modulation model."""
        self.a_long = a_long
        self.k_long = k_long
        self.grav0 = grav0
        self.a_short = a_short
        self.k_short = k_short
        self.grid_size = grid_size
        self.dt = time_step
        self.num_periods = num_periods
        self.omega_long = angular_frequency(self.grav0, self.k_long, 0)
        self.T_long = 2 * np.pi / self.omega_long
        self.phase = np.linspace(0, 2 * np.pi, self.grid_size, endpoint=False)
        self.x = self.phase / self.k_long
        self.dx = self.x[1] - self.x[0]
        self.ds = self.dx * np.ones(
            self.grid_size
        )  # surface-following horizontal coordinate
        self.time = np.arange(0, self.num_periods * self.T_long + self.dt, self.dt)
        self.num_time_steps = len(self.time)
        self.curvilinear = curvilinear

    def get_elevation_ramp(self, t: float) -> float:
        """Determine the long-wave profile."""
        if self.ramp_type == None:
            eta_ramp = 1
        elif self.ramp_type == "linear":
            eta_ramp = t / (self.num_ramp_periods + 1e-10) / self.T_long
            if t > self.num_ramp_periods * self.T_long:
                eta_ramp = 1
        elif self.ramp_type == "groups":
            group_duration = self.num_waves_in_group * self.T_long
            eta_ramp = np.sin(t / group_duration * np.pi) ** 2
        return eta_ramp

    def run(
        self,
        wave_type: str = "linear",
        ramp_type: str = None,
        num_ramp_periods: int = 5,
        num_waves_in_group: int = 10,
        save_tendencies: bool = False,
    ):
        """Integrate the model forward in time."""
        if wave_type not in ["linear", "stokes"]:
            raise ValueError("Invalid wave_type")

        self.elevation = elevation
        self.gravity = gravity_curvilinear
        self._wave_type = wave_type

        if not ramp_type in [None, "linear", "groups"]:
            raise ValueError("Invalid ramp_type")

        self.ramp_type = ramp_type
        self.num_ramp_periods = num_ramp_periods
        self.num_waves_in_group = num_waves_in_group
        self.save_tendencies = save_tendencies

        # Allocate and initialize prognostic fields.
        self.k = np.zeros((self.num_time_steps, self.grid_size), dtype=np.float32)
        self.a = np.zeros_like(self.k)
        self.N = np.zeros_like(self.k)
        self.k[0] = self.k_short
        self.a[0] = self.a_short
        self.N[0] = 1  # FIXME: This is a placeholder value.

        # Allocate and initialize short-wave diagnostic fields.
        self.g = np.zeros_like(self.k)
        self.omega = np.zeros_like(self.k)

        # Allocate tendencies if requested.
        if self.save_tendencies:
            self.k_propagation_tendency = np.zeros_like(self.k)
            self.k_advection_tendency = np.zeros_like(self.k)
            self.k_convergence_tendency = np.zeros_like(self.k)
            self.k_inhomogeneity_tendency = np.zeros_like(self.k)
            self.N_propagation_tendency = np.zeros_like(self.N)
            self.N_advection_tendency = np.zeros_like(self.N)
            self.N_convergence_tendency = np.zeros_like(self.N)
            self.N_inhomogeneity_tendency = np.zeros_like(self.N)

        self.current_time_step = 0

        for n in track(
            range(1, self.num_time_steps),
            description="Running the model",
            total=self.num_time_steps - 1,
            auto_refresh=False,
        ):
            self.current_time_step = n - 1
            self.k[n] = rk4(
                self.wavenumber_tendency, self.k[n - 1], self.time[n - 1], self.dt
            )
            self.N[n] = rk4(
                self.waveaction_tendency, self.N[n - 1], self.time[n - 1], self.dt
            )

        # Fill in the last time step.
        self.g[-1] = self.g[-2]
        self.omega[-1] = self.omega[-2]

        omega0 = angular_frequency(self.grav0, self.k_short)
        self.a = self.a_short * np.sqrt(
            self.grav0 * self.N * self.omega / (omega0 * self.g)
        )

    def wavenumber_tendency(self, k, t):
        """Compute the tendencies of the wavenumber conservation balance at time t."""
        eta_ramp = self.get_elevation_ramp(t)

        eta = eta_ramp * self.elevation(
            self.x, t, self.a_long, self.k_long, self.omega_long, self._wave_type
        )

        if self.curvilinear:
            slope = eta_ramp * surface_slope(
                self.x, t, self.a_long, self.k_long, self.omega_long, self._wave_type
            )
            alpha = np.arctan(slope)
            self.ds = self.dx / np.cos(alpha)
            u = orbital_horizontal_velocity(
                self.x, eta, t, eta_ramp * self.a_long, self.k_long, self.omega_long
            )
            w = orbital_vertical_velocity(
                self.x, eta, t, eta_ramp * self.a_long, self.k_long, self.omega_long
            )
            vel = u * np.cos(alpha) + w * np.sin(alpha)
        else:
            self.ds = self.dx * np.ones(self.grid_size)
            vel = orbital_horizontal_velocity(
                self.x, eta, t, eta_ramp * self.a_long, self.k_long, self.omega_long
            )

        g = self.gravity(
            self.x,
            t,
            eta_ramp * self.a_long,
            self.k_long,
            self.omega_long,
            self.grav0,
            wave_type=self._wave_type,
        )
        self.g[self.current_time_step] = g

        omega = angular_frequency(g, k)
        self.omega[self.current_time_step] = omega

        Cg = omega / k / 2
        k_propagation_tendency = -Cg * diff(k) / self.ds
        k_advection_tendency = -vel * diff(k) / self.ds
        k_convergence_tendency = -k * diff(vel) / self.ds
        k_inhomogeneity_tendency = -0.5 * np.sqrt(k / g) * diff(g) / self.ds
        res = (
            k_propagation_tendency
            + k_advection_tendency
            + k_convergence_tendency
            + k_inhomogeneity_tendency
        )
        if self.save_tendencies:
            self.k_propagation_tendency[self.current_time_step] = k_propagation_tendency
            self.k_advection_tendency[self.current_time_step] = k_advection_tendency
            self.k_convergence_tendency[self.current_time_step] = k_convergence_tendency
            self.k_inhomogeneity_tendency[self.current_time_step] = (
                k_inhomogeneity_tendency
            )
        return res

    def waveaction_tendency(self, N, t):
        """Compute the tendencies of the wave action balance at time t."""
        eta_ramp = self.get_elevation_ramp(t)

        eta = eta_ramp * self.elevation(
            self.x, t, self.a_long, self.k_long, self.omega_long, self._wave_type
        )

        if self.curvilinear:
            slope = eta_ramp * surface_slope(
                self.x, t, self.a_long, self.k_long, self.omega_long, self._wave_type
            )
            alpha = np.arctan(slope)
            self.ds = self.dx / np.cos(alpha)
            u = orbital_horizontal_velocity(
                self.x, eta, t, eta_ramp * self.a_long, self.k_long, self.omega_long
            )
            w = orbital_vertical_velocity(
                self.x, eta, t, eta_ramp * self.a_long, self.k_long, self.omega_long
            )
            vel = u * np.cos(alpha) + w * np.sin(alpha)
        else:
            self.ds = self.dx * np.ones(self.grid_size)
            vel = orbital_horizontal_velocity(
                self.x, eta, t, eta_ramp * self.a_long, self.k_long, self.omega_long
            )

        g = self.gravity(
            self.x,
            t,
            eta_ramp * self.a_long,
            self.k_long,
            self.omega_long,
            self.grav0,
            wave_type=self._wave_type,
        )
        Cg = (
            angular_frequency(g, self.k[self.current_time_step])
            / self.k[self.current_time_step]
            / 2
        )
        N_propagation_tendency = -Cg * diff(N) / self.ds
        N_advection_tendency = -vel * diff(N) / self.ds
        N_convergence_tendency = -N * diff(vel) / self.ds
        N_inhomogeneity_tendency = -N * diff(Cg) / self.ds
        if self.save_tendencies:
            self.N_propagation_tendency[self.current_time_step] = N_propagation_tendency
            self.N_advection_tendency[self.current_time_step] = N_advection_tendency
            self.N_convergence_tendency[self.current_time_step] = N_convergence_tendency
            self.N_inhomogeneity_tendency[self.current_time_step] = (
                N_inhomogeneity_tendency
            )
        res = (
            N_propagation_tendency
            + N_advection_tendency
            + N_convergence_tendency
            + N_inhomogeneity_tendency
        )
        return res

    def to_xarray(self) -> xr.Dataset:
        """Convert the model output to an xarray dataset."""
        ds = xr.Dataset(
            {
                "wavenumber": (("time", "space"), self.k),
                "amplitude": (("time", "space"), self.a),
                "wave_action": (("time", "space"), self.N),
                "gravitational_acceleration": (("time", "space"), self.g),
                "angular_frequency": (("time", "space"), self.omega),
            },
            coords={"time": self.time, "space": self.x},
        )
        if self.save_tendencies:
            ds["k_propagation_tendency"] = (
                ("time", "space"),
                self.k_propagation_tendency,
            )
            ds["k_advection_tendency"] = (("time", "space"), self.k_advection_tendency)
            ds["k_convergence_tendency"] = (
                ("time", "space"),
                self.k_convergence_tendency,
            )
            ds["k_inhomogeneity_tendency"] = (
                ("time", "space"),
                self.k_inhomogeneity_tendency,
            )
            ds["N_propagation_tendency"] = (
                ("time", "space"),
                self.N_propagation_tendency,
            )
            ds["N_advection_tendency"] = (("time", "space"), self.N_advection_tendency)
            ds["N_convergence_tendency"] = (
                ("time", "space"),
                self.N_convergence_tendency,
            )
            ds["N_inhomogeneity_tendency"] = (
                ("time", "x"),
                self.N_inhomogeneity_tendency,
            )
        return ds

    def __repr__(self) -> str:
        return f"WaveModulationModel(a_long={self.a_long}, k_long={self.k_long}, grav0={self.grav0})"
