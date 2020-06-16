import pytest

from ....thirdparty_adapters import to_output_type
from .test_utils import small_clf_dataset, assert_array_equal
from .._data import PolynomialFeatures
import numpy as np
import cupy as cp

import operator as op
from functools import reduce


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("order", ['C', 'F'])
def test_poly_features(small_clf_dataset, degree, interaction_only,
                       include_bias, order):
    (np_X, np_y), (X, y) = small_clf_dataset

    polyfeatures = PolynomialFeatures(degree=degree, order=order,
                                      interaction_only=interaction_only,
                                      include_bias=include_bias)
    transformed_X = polyfeatures.fit_transform(X)
    assert str(type(X)) == str(type(transformed_X))

    if isinstance(transformed_X, np.ndarray):
        if order == 'C':
            assert transformed_X.flags['C_CONTIGUOUS']
        elif order == 'F':
            assert transformed_X.flags['F_CONTIGUOUS']

    transformed_X = to_output_type(transformed_X, 'numpy')

    n_features = np_X.shape[1]

    start = 0 if include_bias else 1
    n_combinations = sum(ncr(n_features, i) for i in range(start, degree+1))

    n_outputs = transformed_X.shape[1]
    if interaction_only:
        assert n_outputs == n_combinations
    else:
        assert n_outputs > n_combinations
