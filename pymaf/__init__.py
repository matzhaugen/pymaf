from numpy.linalg import svd
from numpy import diff, cov, dot, diag, power, sqrt, sum, fliplr, array, asarray

def maf(x):
	"""Compute the maf factors of a set of time series


	Input: x(np.array/pd.dataframe) dimensions n-by-p with p n-length time series
	Output: f(np.array) Maf factors, i.e. the rotated time series such that the first column
	is the linear combination of the original time series that maximizes autocorrelation. The 
	second column maximizes autocorrelation and is also orthogonal to the first time series.
		w (p-by-p np.array) - maf weights, or the loadings of each original time series. 
		Each column represents a maf factor. 
	"""
	x = asarray(x)
	n, p = x.shape
	if n - 1 < p:
	    raise ValueError("Not enough observations to estimate weights")
	s = cov(x.T)
	u, d, v = svd(s)
	a = dot(dot(u, diag(power(d, -0.5))), u.T)
	y = dot(x, a)
	dy = diff(y, n=1, axis=0)
	sy = cov(dy.T)
	uy, dy, vy = svd(sy)
	w = fliplr(dot(a, uy))
	w = array([wi/sqrt(sum(wi**2)) for wi in w.T]).T
	f = dot(x, w)

	return f, w 
