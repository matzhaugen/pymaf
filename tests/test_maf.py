import pytest
import numpy as np
from pymaf import maf, windowed_maf
from test_data import test_data


def test_maf():
    n = test_data.shape[0]
    mafs, coefs = maf(test_data)
    assert np.all(np.sum(np.diff(mafs, axis=0), axis=0) > 0)
    np.testing.assert_almost_equal(np.sum(coefs ** 2, axis=0), 1)


@pytest.mark.parametrize("min_obs, steps", [(50, 2), (100, 9)])
def test_windowed_maf(min_obs, steps):
    n = test_data.shape[0]
    mafs, coefs = windowed_maf(test_data, min_observations=min_obs, steps=steps)
    np.testing.assert_almost_equal(np.sum(coefs ** 2, axis=1), 1)
