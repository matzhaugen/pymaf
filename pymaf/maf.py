from numpy.linalg import svd
import numpy as np


def scale(arr):
    return np.apply_along_axis(lambda x: (x - np.mean(x)) / np.std(x), arr=arr, axis=0)


def maf(x):
    """Compute the maf factors of a set of time series


    Input: x(np.array/pd.dataframe) dimensions n-by-p with p n-length time series
    Output: f(np.array) Maf factors, i.e. the rotated time series such that the first column
    is the linear combination of the original time series that maximizes autocorrelation. The
    second column maximizes autocorrelation and is also orthogonal to the first time series.
            w (p-by-p np.array) - maf weights, or the loadings of each original time series.
        Each column represents a maf factor.
    """
    x = np.asarray(x)
    n, p = x.shape
    if n - 1 < p:
        raise ValueError("Not enough observations to estimate weights")
    s = np.cov(x.T)
    u, d, v = svd(s)
    a = np.dot(np.dot(u, np.diag(np.power(d, -0.5))), u.T)
    y = np.dot(x, a)
    dy = np.diff(y, n=1, axis=0)
    sy = np.cov(dy.T)
    uy, dy, vy = svd(sy)
    w = np.fliplr(np.dot(a, uy))
    w = np.array([wi / np.sqrt(sum(wi**2)) for wi in w.T]).T

    f = np.dot(x, w)

    # We disambiguate the mafs by correlating the maf factors with a linear function
    # with time, i.e. 1:t where t is the number of time steps.
    flip = np.sum(np.diff(f, axis=0), axis=0) > 0
    flip = flip.astype(int) * 2 - 1
    w = w * flip[np.newaxis, :]
    f = np.dot(x, w)

    return f, w


def windowed_maf(x: np.ndarray, min_observations: int, steps: int):
    """(Experimental feature): Compute the maf over a moving time window. If the number of
    observations are not large enough, the weights will be unstable and misaligned with other time windows
    leading to inconsistent mafs. In this case, it is wise to verify that the coefficients produce reasonable results
    e.g. by correlating the resultant mafs with time or some other measure of consistency.

    Args:
        x (np.ndarray): Input data, with each column being a time series, each row is a multivariate timeseries observation.
        min_observations (int): Minimum number of observations to include in the computing of the maf weights/factors.
        steps (int): The steps to increment

    Returns:
        np.ndarray: Maf factors, i.e. the rotated time series such that the first column
        is the linear combination of the original time series that maximizes autocorrelation. The
        second column maximizes autocorrelation and is also orthogonal to the first time series.
        np.ndarray: A 3d array with number of observations, n, in the first dimension, the loadings on each predictor in the second,
        and the MAF number in the third, i.e. n x p x p where p is the number of time series.

    """
    x = np.asarray(x)

    n, p = x.shape
    assert min_observations >= p + 1, "The minimum observations must be greater than the number of time series"
    weights = []
    start_idx = np.arange(0, n - min_observations + 1, steps)
    end_idx = np.arange(min_observations, n + 1, steps)

    weights = np.zeros((len(start_idx), p, p))
    for i, (start, end) in enumerate(zip(start_idx, end_idx)):
        chunk = x[start:end, :]
        f, weights[i, :, :] = maf(chunk)

    windowed_weights = np.zeros((n, p, p))
    last_end = 0
    for i, (start, end) in enumerate(zip(start_idx, end_idx)):
        windowed_weights[last_end:end, :, :] = weights[i, :, :]
        last_end = end
    if end < n:
        windowed_weights[end:, :, :] = windowed_weights[(end - 1), :, :]

    windowed_mafs = windowed_weights * x[:, :, np.newaxis]

    return windowed_mafs, windowed_weights
