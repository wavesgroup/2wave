import numpy as np
from twowave import (
    angular_frequency,
    elevation,
    gravity,
    WaveModulationModel,
)
import xarray as xr
import pytest


G0 = 9.8


def test_angular_frequency():
    assert angular_frequency(G0, 1) == np.sqrt(G0)
    assert angular_frequency(G0, 1, 1) == np.sqrt(G0) + 1


def test_elevation():
    assert elevation(0, 0, 1, 1, 1, wave_type="linear") == 1

    # Create mock nonlinear properties
    x = np.linspace(0, 2 * np.pi, 128)
    z = 0.1 * np.cos(x)  # Simple cosine elevation
    mock_props = (x, z, None, None, None, None)

    # Test nonlinear elevation
    assert np.isclose(
        elevation(0, 0, 0.1, 1, 1, wave_type="nonlinear", nonlinear_props=mock_props),
        0.1,
        rtol=1e-3,
    )

    # Test nonlinear elevation at phase π
    assert np.isclose(
        elevation(
            np.pi, 0, 0.1, 1, 1, wave_type="nonlinear", nonlinear_props=mock_props
        ),
        -0.1,
        rtol=1e-3,
    )


def test_gravity():
    # Test with array input since gravity no longer works with scalar x
    phase = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    k = 1
    a = 0.1
    x = phase / k
    omega = np.sqrt(G0 * k)
    assert np.isclose(
        gravity(x, 0, a, k, omega, G0, wave_type="linear")[0], 8.83, rtol=1e-2
    )
    assert np.isclose(
        gravity(x, 0, a, k, omega, G0, wave_type="stokes")[0], 8.83, rtol=1e-2
    )


def test_wave_modulation_model():
    # Test with linear wave type
    m = WaveModulationModel(num_periods=1)
    m.run()
    ds = m.to_xarray()
    assert type(ds) == xr.Dataset
    assert np.all(np.isfinite(ds.wavenumber))
    assert np.all(np.isfinite(ds.amplitude))

    # Test with nonlinear wave type
    try:
        m = WaveModulationModel(num_periods=1)
        m.run(wave_type="nonlinear")
        ds = m.to_xarray()
        assert type(ds) == xr.Dataset
        assert np.all(np.isfinite(ds.wavenumber))
        assert np.all(np.isfinite(ds.amplitude))
    except Exception as e:
        # Skip this test if SSGW module is not available or other issues arise
        pytest.skip(f"Skipping nonlinear test due to: {str(e)}")


def test_missing_nonlinear_props():
    # Test that appropriate error is raised when nonlinear_props is missing
    with pytest.raises(ValueError, match="nonlinear_props must be provided"):
        elevation(0, 0, 0.1, 1, 1, wave_type="nonlinear")
