import numpy as np


test_data = np.array(
    [
        np.arange(100) + np.random.normal(size=100) * 10,
        1 + np.random.normal(size=100),
        np.arange(100, 0, -1) + np.random.normal(size=100) * 5,
    ]
).T
