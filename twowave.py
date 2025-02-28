"""
A two-wave hydrodynamic modulation model.
"""

import numpy as np
from rich.progress import track
from scipy.interpolate import interp1d
from ssgw import SSGW
import xarray as xr


def angular_frequency(g: float, k: float, u: float = 0) -> float:
    return np.sqrt(g * k) + k * u


def elevation(
    x: float,
    t: float,
    a: float,
    k: float,
    omega: float,
    wave_type: str = "linear",
    nonlinear_props: tuple = None,
) -> float:
    phase = k * x - omega * t
    if wave_type == "linear":
        return a * np.cos(phase)
    elif wave_type == "stokes":
        term1 = np.cos(phase)
        term2 = 0.5 * a * k * np.cos(2 * phase)
        term3 = (a * k) ** 2 * (3 / 8 * np.cos(3 * phase) - 1 / 16 * np.cos(phase))
        return a * (term1 + term2 + term3)
    elif wave_type == "nonlinear":
        if nonlinear_props is None:
            raise ValueError("nonlinear_props must be provided for nonlinear wave type")
        positions, elevations, _, _, _, _ = nonlinear_props

        # Handle both scalar and array inputs
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)

        # Apply for each x value
        for i, xi in enumerate(x_array):
            # Apply phase shift based on time
            x_phase = (xi - omega / k * t) % (2 * np.pi / k)
            # Interpolate elevation at the phase-shifted position
            result[i] = np.interp(x_phase * k, positions, elevations, period=2 * np.pi)

        # Return scalar if input was scalar
        return result[0] if np.isscalar(x) else result
    else:
        raise ValueError("wave_type must be 'linear', 'stokes', or 'nonlinear'")


def surface_slope(
    x: float,
    t: float,
    a: float,
    k: float,
    omega: float,
    wave_type: str = "linear",
    nonlinear_props: tuple = None,
) -> float:
    phase = k * x - omega * t
    ak = a * k
    if wave_type == "linear":
        slope = -ak * np.sin(phase)
        return slope
    elif wave_type == "stokes":
        term1 = -ak * np.sin(phase)
        term2 = -(ak**2) * np.sin(2 * phase)
        term3 = -(ak**3) * (9 / 8 * np.sin(3 * phase) - 1 / 16 * np.sin(phase))
        slope = term1 + term2 + term3
        return slope
    elif wave_type == "nonlinear":
        if nonlinear_props is None:
            raise ValueError("nonlinear_props must be provided for nonlinear wave type")
        positions, elevations, _, _, _, _ = nonlinear_props

        # Handle both scalar and array inputs
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)

        # Calculate dense slopes once
        dense_pos = np.linspace(0, 2 * np.pi, 1000)
        dense_elev = np.interp(dense_pos, positions, elevations, period=2 * np.pi)
        dx = dense_pos[1] - dense_pos[0]
        slopes = np.gradient(dense_elev, dx)

        # Apply for each x value
        for i, xi in enumerate(x_array):
            # Apply phase shift based on time
            x_phase = (xi - omega / k * t) % (2 * np.pi / k)
            # Interpolate slope at the requested position
            result[i] = np.interp(x_phase * k, dense_pos, slopes, period=2 * np.pi) / k

        # Return scalar if input was scalar
        return result[0] if np.isscalar(x) else result
    else:
        raise ValueError("wave_type must be 'linear', 'stokes', or 'nonlinear'")


def orbital_horizontal_velocity(
    x: float,
    z: float,
    t: float,
    a: float,
    k: float,
    omega: float,
    wave_type: str = "linear",
    nonlinear_props: tuple = None,
) -> float:
    """Horizontal orbital velocity at depth z.
    Supports linear, Stokes, and nonlinear wave types.
    """
    if wave_type in ["linear", "stokes"]:
        return a * omega * np.cos(k * x - omega * t) * np.exp(k * z)
    elif wave_type == "nonlinear":
        if nonlinear_props is None:
            raise ValueError("nonlinear_props must be provided for nonlinear wave type")
        positions, _, u, _, _, _ = nonlinear_props

        # Handle both scalar and array inputs
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)

        # Apply for each x value
        for i, xi in enumerate(x_array):
            # Apply phase shift based on time
            x_phase = (xi - omega / k * t) % (2 * np.pi / k)
            # Get horizontal velocity at this position
            result[i] = np.interp(x_phase * k, positions, u, period=2 * np.pi)

        # Return scalar if input was scalar
        return result[0] if np.isscalar(x) else result
    else:
        raise ValueError("wave_type must be 'linear', 'stokes', or 'nonlinear'")


def orbital_vertical_velocity(
    x: float,
    z: float,
    t: float,
    a: float,
    k: float,
    omega: float,
    wave_type: str = "linear",
    nonlinear_props: tuple = None,
) -> float:
    """Vertical orbital velocity at depth z.
    Supports linear, Stokes, and nonlinear wave types.
    """
    if wave_type in ["linear", "stokes"]:
        return a * omega * np.sin(k * x - omega * t) * np.exp(k * z)
    elif wave_type == "nonlinear":
        if nonlinear_props is None:
            raise ValueError("nonlinear_props must be provided for nonlinear wave type")
        positions, _, _, w, _, _ = nonlinear_props

        # Handle both scalar and array inputs
        x_array = np.atleast_1d(x)
        result = np.zeros_like(x_array, dtype=float)

        # Apply for each x value
        for i, xi in enumerate(x_array):
            # Apply phase shift based on time
            x_phase = (xi - omega / k * t) % (2 * np.pi / k)
            # Get vertical velocity at this position
            result[i] = np.interp(x_phase * k, positions, w, period=2 * np.pi)

        # Return scalar if input was scalar
        return result[0] if np.isscalar(x) else result
    else:
        raise ValueError("wave_type must be 'linear', 'stokes', or 'nonlinear'")


def gravity(
    x: float,
    t: float,
    a: float,
    k: float,
    omega: float,
    g0: float = 9.8,
    wave_type: str = "linear",
    nonlinear_props: tuple = None,
    curvilinear: bool = True,
) -> float:
    z = elevation(x, t, a, k, omega, wave_type, nonlinear_props)
    U = orbital_horizontal_velocity(x, z, t, a, k, omega, wave_type, nonlinear_props)
    W = orbital_vertical_velocity(x, z, t, a, k, omega, wave_type, nonlinear_props)
    Cp = omega / k
    dx = np.diff(x)[0]
    a_z = -diff(W) / dx * (Cp - U)
    if curvilinear:
        a_x = -diff(U) / dx * (Cp - U)
        slope = surface_slope(x, t, a, k, omega, wave_type, nonlinear_props)
        g = g0 * np.cos(slope) + a_z * np.cos(slope) + a_x * np.sin(slope)
    else:
        g = g0 + a_z
    return g


def nonlinear_wave_properties(
    a: float, k: float, g0: float = 9.8, num_points: int = 128
):
    ak = a * k
    omega = np.sqrt(g0 * k)  # deep water
    wave = SSGW(np.inf, ak, 128)
    Cp = wave.ce * omega / k
    x, z = wave.zs.real, wave.zs.imag
    u, w = (wave.ws.real + wave.ce) * np.sqrt(g0), -wave.ws.imag * np.sqrt(g0)

    # Shift the wave solution by pi
    x -= np.pi
    z = np.array(z[len(z) // 2 :].tolist() + z[: len(z) // 2].tolist())
    u = np.array(u[len(u) // 2 :].tolist() + u[: len(u) // 2].tolist())
    w = np.array(w[len(w) // 2 :].tolist() + w[: len(w) // 2].tolist())

    dx = diff(x)
    dx[dx < 0] += np.pi
    a_z = -diff(w) / dx * (Cp - u)
    a_x = -diff(u) / dx * (Cp - u)

    # Shift back by pi
    x += np.pi
    z = np.array(z[len(z) // 2 :].tolist() + z[: len(z) // 2].tolist())
    u = np.array(u[len(u) // 2 :].tolist() + u[: len(u) // 2].tolist())
    w = np.array(w[len(w) // 2 :].tolist() + w[: len(w) // 2].tolist())
    a_x = np.array(a_x[len(a_x) // 2 :].tolist() + a_x[: len(a_x) // 2].tolist())
    a_z = np.array(a_z[len(a_z) // 2 :].tolist() + a_z[: len(a_z) // 2].tolist())

    x_ = np.linspace(x[0], x[-1], num_points)
    z = interp1d(x, z, kind="cubic")(x_)
    u = interp1d(x, u, kind="cubic")(x_)
    w = interp1d(x, w, kind="cubic")(x_)
    a_x = interp1d(x, a_x, kind="cubic")(x_)
    a_z = interp1d(x, a_z, kind="cubic")(x_)

    return x_, z, u, w, a_x, a_z


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
        grid_size: int = 128,
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
        self.num_periods = num_periods
        self.omega_long = angular_frequency(self.grav0, self.k_long, 0)
        self.T_long = 2 * np.pi / self.omega_long
        self.phase = np.linspace(0, 2 * np.pi, self.grid_size, endpoint=False)
        self.x = self.phase / self.k_long
        self.dx = self.x[1] - self.x[0]
        self.ds = self.dx * np.ones(self.grid_size)
        Cg_short = 0.5 * angular_frequency(grav0, k_short) / k_short
        self.dt = self.dx / (Cg_short + self.a_long * self.omega_long)  # CFL
        self.time = np.arange(0, self.num_periods * self.T_long + self.dt, self.dt)
        self.num_time_steps = len(self.time)
        self.curvilinear = curvilinear

        # Initialize nonlinear wave properties as None
        self.nonlinear_props = None

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
        return max(eta_ramp, 1e-6)

    def run(
        self,
        wave_type: str = "linear",
        ramp_type: str = None,
        num_ramp_periods: int = 5,
        num_waves_in_group: int = 10,
        save_tendencies: bool = False,
    ):
        """Integrate the model forward in time."""
        if wave_type not in ["linear", "stokes", "nonlinear"]:
            raise ValueError(
                "Invalid wave_type, must be 'linear', 'stokes', or 'nonlinear'"
            )

        self.elevation = elevation
        self.gravity = gravity
        self._wave_type = wave_type

        if wave_type == "nonlinear":
            positions, elevations, u, w, ax, az = nonlinear_wave_properties(
                self.a_long, self.k_long, self.grav0
            )
            self.nonlinear_props = (positions, elevations, u, w, ax, az)

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

        # Allocate and initialize surface velocities, elevation, and slope.
        self.u = np.zeros_like(self.k)
        self.w = np.zeros_like(self.k)
        self.eta = np.zeros_like(self.k)
        self.slope = np.zeros_like(self.k)

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

        if self._wave_type == "nonlinear":
            self.nonlinear_props = nonlinear_wave_properties(
                eta_ramp * self.a_long, self.k_long, self.grav0
            )

        eta = self.elevation(
            self.x,
            t,
            eta_ramp * self.a_long,
            self.k_long,
            self.omega_long,
            self._wave_type,
            self.nonlinear_props,
        )
        self.eta[self.current_time_step] = eta

        if self.curvilinear:
            slope = surface_slope(
                self.x,
                t,
                eta_ramp * self.a_long,
                self.k_long,
                self.omega_long,
                self._wave_type,
                self.nonlinear_props,
            )
            self.slope[self.current_time_step] = slope

            alpha = np.arctan(slope)
            self.ds = self.dx / np.cos(alpha)
            u = orbital_horizontal_velocity(
                self.x,
                eta,
                t,
                eta_ramp * self.a_long,
                self.k_long,
                self.omega_long,
                wave_type=self._wave_type,
                nonlinear_props=self.nonlinear_props,
            )
            self.u[self.current_time_step] = u

            w = orbital_vertical_velocity(
                self.x,
                eta,
                t,
                eta_ramp * self.a_long,
                self.k_long,
                self.omega_long,
                wave_type=self._wave_type,
                nonlinear_props=self.nonlinear_props,
            )
            self.w[self.current_time_step] = w

            vel = u * np.cos(alpha) + w * np.sin(alpha)
        else:
            self.ds = self.dx * np.ones(self.grid_size)
            vel = orbital_horizontal_velocity(
                self.x,
                eta,
                t,
                eta_ramp * self.a_long,
                self.k_long,
                self.omega_long,
                wave_type=self._wave_type,
                nonlinear_props=self.nonlinear_props,
            )
            self.u[self.current_time_step] = vel

        g = self.gravity(
            self.x,
            t,
            eta_ramp * self.a_long,
            self.k_long,
            self.omega_long,
            self.grav0,
            wave_type=self._wave_type,
            nonlinear_props=self.nonlinear_props,
            curvilinear=self.curvilinear,
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

        if self._wave_type == "nonlinear":
            self.nonlinear_props = nonlinear_wave_properties(
                eta_ramp * self.a_long, self.k_long, self.grav0
            )

        eta = self.elevation(
            self.x,
            t,
            eta_ramp * self.a_long,
            self.k_long,
            self.omega_long,
            self._wave_type,
            self.nonlinear_props,
        )

        if self.curvilinear:
            slope = surface_slope(
                self.x,
                t,
                eta_ramp * self.a_long,
                self.k_long,
                self.omega_long,
                self._wave_type,
                self.nonlinear_props,
            )
            alpha = np.arctan(slope)
            self.ds = self.dx / np.cos(alpha)
            u = orbital_horizontal_velocity(
                self.x,
                eta,
                t,
                eta_ramp * self.a_long,
                self.k_long,
                self.omega_long,
                wave_type=self._wave_type,
                nonlinear_props=self.nonlinear_props,
            )
            w = orbital_vertical_velocity(
                self.x,
                eta,
                t,
                eta_ramp * self.a_long,
                self.k_long,
                self.omega_long,
                wave_type=self._wave_type,
                nonlinear_props=self.nonlinear_props,
            )
            vel = u * np.cos(alpha) + w * np.sin(alpha)
        else:
            self.ds = self.dx * np.ones(self.grid_size)
            vel = orbital_horizontal_velocity(
                self.x,
                eta,
                t,
                eta_ramp * self.a_long,
                self.k_long,
                self.omega_long,
                wave_type=self._wave_type,
                nonlinear_props=self.nonlinear_props,
            )

        g = self.gravity(
            self.x,
            t,
            eta_ramp * self.a_long,
            self.k_long,
            self.omega_long,
            self.grav0,
            wave_type=self._wave_type,
            nonlinear_props=self.nonlinear_props,
            curvilinear=self.curvilinear,
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
                "elevation": (("time", "space"), self.eta),
                "slope": (("time", "space"), self.slope),
                "horizontal_velocity": (("time", "space"), self.u),
                "vertical_velocity": (("time", "space"), self.w),
                "surface_slope": (("time", "space"), self.slope),
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
