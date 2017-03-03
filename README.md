# pymaf

The Python version of the MAF algorithm, which takes a set of concurrent time series and extracts the linear combinations that maximize the autocorrelation of the time series. The first MAF time series maximizes the autocorrelation without any other constraints, while each subsequent MAF has the additional contraint that it is orthogonal to the previous MAF time series. Orthogonal means that the vector dot product is zero. 

## Install

To install
Open the terminal, go to the pymaf directory, and type
```
pip install -e .
```

Then open a python shell and type 
```
from pymaf import maf
```
