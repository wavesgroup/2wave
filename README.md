# 2wave

A hydrodynamic modulation model for surface waves.

## Getting started

### Get the code and install dependencies

```
git clone https://github.com/wavesgroup/2wave
cd 2wave
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the model

```python
from model import WaveModulationModel

m = WaveModulationModel()  # import the model
m.run()  # run the model
ds = m.to_xarray()  # convert the model output to an xarray dataset
```

## Questions or issues?

Open a [new issue](https://github.com/wavesgroup/2wave/issues/new) on GitHub.