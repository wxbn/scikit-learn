import pytest

from ....thirdparty_adapters import to_output_type
from .test_utils import small_int_dataset, assert_array_equal
from ..imputation import SimpleImputer
import numpy as np
import cupy as cp


@pytest.mark.parametrize("strategy", ["mean", "most_frequent", "constant"])
def test_imputer(small_int_dataset, strategy):
    np_X, X = small_int_dataset
    fill_value = np.random.randint(10, size=1)[0]

    imputer = SimpleImputer(copy=True, strategy=strategy,
                           fill_value=fill_value)
    transformed_X = imputer.fit_transform(X)
    assert str(type(X)) == str(type(transformed_X))
    
    transformed_X = to_output_type(transformed_X, 'numpy')

    t_np_X = np.array(np_X, copy=True)
    n_features = t_np_X.shape[1]

    if strategy == "mean":
        mean = np.nanmean(t_np_X, axis=0)
        for i in range(n_features):
            mask = np.where(np.isnan(t_np_X[:, i]))
            t_np_X[mask, i] = mean[i]
    elif strategy == "most_frequent":
        for i in range(n_features):
            values, counts = np.unique(t_np_X[:, i], return_counts=True)
            max_idx = np.argmax(counts)
            most_frequent = values[max_idx]

            mask = np.where(np.isnan(t_np_X[:, i]))
            t_np_X[mask, i] = most_frequent
    elif strategy == "constant":
        t_np_X[np.where(np.isnan(t_np_X))] = fill_value

    transformed_np_X = t_np_X
    assert not np.isnan(transformed_np_X).any()

    assert_array_equal(transformed_X, transformed_np_X, mean_diff_tol=0.0001, max_diff_tol=0.0001)
