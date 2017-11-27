# pymaf

The Python version of the MAF algorithm, which takes a set of concurrent time series and extracts the linear combinations that maximize the autocorrelation of the time series. The first MAF time series maximizes the autocorrelation without any other constraints, while each subsequent MAF has the additional contraint that it is orthogonal to the previous MAF time series. Two time series are orthogonal if the vector dot product is zero. 

## Install

To install
Open the terminal, and type
```
pip install pymaf
```

Then open a python shell and type 
```
from pymaf import maf
```

## Example
Here is an example of some timeseries
```
# Import libraries
from pymaf import maf
import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt

# Import data
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2013, 1, 27)
prices = web.DataReader(["F", "AAPL", "GOOG", "AMD", "ACN"], 'google', start, end)
open = prices['Open', :, :].dropna()

# Calculate MAFs and normalize MAF factors
mafs = maf(open)
first_mafs = np.apply_along_axis(lambda x: (x - np.mean(x)) / np.std(x), arr=mafs[0][:,:3], axis=0)

# Plot
labels = ['MAF' + s for s in ['1','2','3'] ]
for y_arr, label in zip(first_mafs.T, labels):
	plt.plot(y_arr, label=label)
plt.legend()
plt.show()
```