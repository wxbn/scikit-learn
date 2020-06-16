import pytest
from ....thirdparty_adapters import to_output_type
from .test_utils import small_clf_dataset, assert_array_equal
from .._data import StandardScaler, MinMaxScaler, MaxAbsScaler
import numpy as np
import cupy as cp


def test_minmax_scaler(small_clf_dataset):
    (np_X, np_y), (X, y) = small_clf_dataset

    scaler = MinMaxScaler(copy=True)
    transformed_X = scaler.fit_transform(X)
    assert str(type(X)) == str(type(transformed_X))

    transformed_X = to_output_type(transformed_X, 'numpy')

    data_min = np.nanmin(np_X, axis=0)
    data_range = np.nanmax(np_X, axis=0) - data_min
    data_range[data_range == 0.0] = 1.0
    scale = 1.0 / data_range
    mini = 0.0 - data_min * scale
    transformed_np_X = np_X * scale + mini

    assert_array_equal(transformed_X, transformed_np_X, mean_diff_tol=0.0001, max_diff_tol=0.0001)

    reversed_X = scaler.inverse_transform(transformed_X)
    assert_array_equal(reversed_X, np_X, mean_diff_tol=0.0001, max_diff_tol=0.0001)


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler(small_clf_dataset, with_mean, with_std):
    (np_X, np_y), (X, y) = small_clf_dataset

    scaler = StandardScaler(copy=True, with_mean=with_mean, with_std=with_std)
    transformed_X = scaler.fit_transform(X)
    assert str(type(X)) == str(type(transformed_X))

    transformed_X = to_output_type(transformed_X, 'numpy')

    t_np_X = np.array(np_X, copy=True)
    if with_mean:
        t_np_X -= t_np_X.mean(axis=0)
    if with_std:
        t_np_X /= t_np_X.std(axis=0)

    transformed_np_X = t_np_X

    assert_array_equal(transformed_X, transformed_np_X, mean_diff_tol=0.0001, max_diff_tol=0.0001)

    reversed_X = scaler.inverse_transform(transformed_X)
    assert_array_equal(reversed_X, np_X, mean_diff_tol=0.0001, max_diff_tol=0.0001)


def test_maxabs_scaler(small_clf_dataset):
    (np_X, np_y), (X, y) = small_clf_dataset

    scaler = MaxAbsScaler(copy=True)
    transformed_X = scaler.fit_transform(X)
    assert str(type(X)) == str(type(transformed_X))
    
    transformed_X = to_output_type(transformed_X, 'numpy')

    max_abs = np.nanmax(np.abs(np_X), axis=0)
    max_abs[max_abs == 0.0] = 1.0
    transformed_np_X = np_X / max_abs

    assert_array_equal(transformed_X, transformed_np_X, mean_diff_tol=0.0001, max_diff_tol=0.0001)

    reversed_X = scaler.inverse_transform(transformed_X)
    assert_array_equal(reversed_X, np_X, mean_diff_tol=0.0001, max_diff_tol=0.0001)
