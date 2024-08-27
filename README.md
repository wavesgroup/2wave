# 2wave

A hydrodynamic modulation model for short surface gravity waves
riding on longer waves.

## Features

* Solves the full wave crest and action balance equations in 1-d.
* Linear (1st order) or Stokes (3rd order) long waves
* Infinite long-wave trains or long-wave groups
* Curvilinear effects on effective gravity of short waves
* Optionally, output all tendencies at all time steps
* Output as Xarray Dataset

## Getting started

### Install 2wave

```
pip install 2wave
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
