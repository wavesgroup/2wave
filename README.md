# 2wave

A hydrodynamic modulation model for short surface gravity waves
riding on longer waves.

## Features

* Solves the nonlinear wave crest and action balance equations in 1-d.
* Wave types:
  - 1st order linear wave
  - 3rd order Stokes wave
  - Fully nonlinear wave using [SSGW](https://github.com/wavesgroup/ssgw) by Clamond & Dutykh (2018, JFM)
* Infinite long-wave trains or long-wave groups
* Effective gravity, propagation, and advection in curvilinear coordinates
* Output as Xarray Dataset
* Optionally, output all tendencies at all time steps

## Getting started

### Install 2wave

```
pip install git+https://github.com/wavesgroup/2wave
```

### Run the model

```python
from twowave import WaveModulationModel

m = WaveModulationModel()  # import the model
m.run()  # run the model
ds = m.to_xarray()  # convert the model output to an xarray dataset
```

### Running the tests

```
git clone https://github.com/wavesgroup/2wave
cd 2wave
python3 -m venv venv
source venv/bin/activate
pip install -U .
pytest
```

## Questions or issues?

Open a [new issue](https://github.com/wavesgroup/2wave/issues/new) on GitHub.
