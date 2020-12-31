import numpy as np
from pymaf import maf
from test_data import test_data, expected_coefs, expected_mafs


def test_maf():
    mafs, coefs = maf(test_data)
    np.testing.assert_almost_equal(coefs, expected_coefs)
    np.testing.assert_almost_equal(mafs, expected_mafs)
