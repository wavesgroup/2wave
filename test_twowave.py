import numpy as np
from twowave import angular_frequency, elevation, gravity, WaveModulationModel
import xarray as xr


G0 = 9.8


def test_angular_frequency():
    assert angular_frequency(G0, 1) == np.sqrt(G0)
    assert angular_frequency(G0, 1, 1) == np.sqrt(G0) + 1


def test_elevation():
    assert elevation(0, 0, 1, 1, 1, wave_type="linear") == 1


def test_gravity():
    assert gravity(0, 0, 0, 1, G0, wave_type="linear") == G0
    assert gravity(0, 0, 0, 1, G0, wave_type="stokes") == G0


def test_wave_modulation_model():
    m = WaveModulationModel(num_periods=1)
    m.run()
    ds = m.to_xarray()
    assert type(ds) == xr.Dataset
    assert np.all(np.isfinite(ds.wavenumber))
    assert np.all(np.isfinite(ds.amplitude))
