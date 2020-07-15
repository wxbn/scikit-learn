# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest

from .sklearn.preprocessing import StandardScaler, MinMaxScaler, \
                    MaxAbsScaler, Normalizer, Binarizer
from .sklearn.preprocessing import scale, minmax_scale, normalize, \
                    add_dummy_feature, binarize
from .sklearn.preprocessing import SimpleImputer
from .sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures as OriginalPolFeat
from sklearn.preprocessing import scale as orignal_scale
from sklearn.preprocessing import add_dummy_feature as original_adf
from sklearn.impute import SimpleImputer as OriginalSimpleImputer

from .test_preproc_utils import assert_allclose
from .test_preproc_utils import clf_dataset  # noqa: F401
from .test_preproc_utils import int_dataset  # noqa: F401
from .test_preproc_utils import sparse_clf_dataset  # noqa: F401
from .test_preproc_utils import sparse_int_dataset  # noqa: F401

import numpy as np
from scipy import sparse as cpu_sp
from cupy import sparse as gpu_sp


def test_minmax_scaler(clf_dataset):  # noqa: F811
    X_np, X = clf_dataset

    scaler = MinMaxScaler(copy=True)
    t_X = scaler.fit_transform(X)
    assert type(t_X) == type(X)

    data_min = np.nanmin(X_np, axis=0)
    data_range = np.nanmax(X_np, axis=0) - data_min
    data_range[data_range == 0.0] = 1.0
    scale = 1.0 / data_range
    mini = 0.0 - data_min * scale
    t_X_np = X_np * scale + mini

    r_X = scaler.inverse_transform(t_X)
    assert type(r_X) == type(t_X)

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)

    assert_allclose(r_X, X_np, rtol=0.0001, atol=0.0001)


def test_minmax_scale(clf_dataset):  # noqa: F811
    X_np, X = clf_dataset

    t_X = minmax_scale(X)
    assert type(t_X) == type(X)

    data_min = np.nanmin(X_np, axis=0)
    data_range = np.nanmax(X_np, axis=0) - data_min
    data_range[data_range == 0.0] = 1.0
    scale = 1.0 / data_range
    mini = 0.0 - data_min * scale
    t_X_np = X_np * scale + mini

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler(clf_dataset, with_mean, with_std):  # noqa: F811
    X_np, X = clf_dataset

    scaler = StandardScaler(copy=True, with_mean=with_mean, with_std=with_std)
    t_X = scaler.fit_transform(X)
    assert type(t_X) == type(X)

    t_X_np = np.array(X_np, copy=True)
    if with_mean:
        t_X_np -= t_X_np.mean(axis=0)
    if with_std:
        t_X_np /= t_X_np.std(axis=0)

    r_X = scaler.inverse_transform(t_X)
    assert type(r_X) == type(t_X)

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)

    assert_allclose(r_X, X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler_sparse(sparse_clf_dataset, with_std):  # noqa: F811
    X_np, X = sparse_clf_dataset

    scaler = StandardScaler(copy=True, with_mean=False, with_std=with_std)
    t_X = scaler.fit_transform(X)
    assert type(t_X) == type(X)

    t_X_np = np.array(X_np, copy=True)
    if with_std:
        t_X_np /= t_X_np.std(axis=0)

    r_X = scaler.inverse_transform(t_X)
    assert type(r_X) == type(t_X)

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)

    assert_allclose(r_X, X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_scale(clf_dataset, with_mean, with_std):  # noqa: F811
    X_np, X = clf_dataset

    t_X = scale(X, copy=True, with_mean=with_mean, with_std=with_std)
    assert type(t_X) == type(X)

    t_X_np = np.array(X_np, copy=True)
    if with_mean:
        t_X_np -= t_X_np.mean(axis=0)
    if with_std:
        t_X_np /= t_X_np.std(axis=0)

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("with_std", [True, False])
def test_scale_sparse(sparse_clf_dataset, with_std):  # noqa: F811
    X_np, X = sparse_clf_dataset

    t_X = scale(X, copy=True, with_mean=False, with_std=with_std)
    assert type(t_X) == type(X)

    X_sp = cpu_sp.csr_matrix(X_np)
    t_X_sp = orignal_scale(X_sp, copy=True, with_mean=False, with_std=with_std)

    t_X_sp = t_X_sp.todense()
    assert_allclose(t_X, t_X_sp, rtol=0.0001, atol=0.0001)


def test_maxabs_scaler(clf_dataset):  # noqa: F811
    X_np, X = clf_dataset

    scaler = MaxAbsScaler(copy=True)
    t_X = scaler.fit_transform(X)
    assert type(t_X) == type(X)

    max_abs = np.nanmax(np.abs(X_np), axis=0)
    max_abs[max_abs == 0.0] = 1.0
    t_X_np = X_np / max_abs

    r_X = scaler.inverse_transform(t_X)
    assert type(r_X) == type(t_X)

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)
    assert_allclose(r_X, X_np, rtol=0.0001, atol=0.0001)


def test_sparse_maxabs_scaler(sparse_clf_dataset):  # noqa: F811
    X_np, X = sparse_clf_dataset

    scaler = MaxAbsScaler(copy=True)
    t_X = scaler.fit_transform(X)
    assert type(t_X) == type(X)

    max_abs = np.nanmax(np.abs(X_np), axis=0)
    max_abs[max_abs == 0.0] = 1.0
    t_X_np = X_np / max_abs

    r_X = scaler.inverse_transform(t_X)
    assert type(r_X) == type(t_X)

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)
    assert_allclose(r_X, X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("norm", ['l1', 'l2', 'max'])
def test_normalizer(sparse_clf_dataset, norm):  # noqa: F811
    X_np, X = sparse_clf_dataset

    if norm == 'l1':
        norms = np.abs(X_np).sum(axis=1)
    elif norm == 'l2':
        norms = np.linalg.norm(X_np, ord=2, axis=1)
    elif norm == 'max':
        norms = np.max(abs(X_np), axis=1)

    t_X_np = np.array(X_np, copy=True)
    t_X_np /= norms[:, np.newaxis]

    try:
        normalizer = Normalizer(norm=norm, copy=True)
        t_X = normalizer.fit_transform(X)
        assert type(t_X) == type(X)

        assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)
    except ValueError:
        pytest.skip("Skipping CSC matrices")


@pytest.mark.parametrize("norm", ['l1', 'l2', 'max'])
@pytest.mark.parametrize("return_norm", [True, False])
def test_normalize(clf_dataset, norm, return_norm):  # noqa: F811
    X_np, X = clf_dataset

    if norm == 'l1':
        norms = np.abs(X_np).sum(axis=0)
    elif norm == 'l2':
        norms = np.linalg.norm(X_np, ord=2, axis=0)
    elif norm == 'max':
        norms = np.max(abs(X_np), axis=0)

    t_X_np = np.array(X_np, copy=True)
    t_X_np /= norms

    if return_norm:
        t_X, t_norms = normalize(X, axis=0, norm=norm, return_norm=return_norm)
        assert_allclose(t_norms, norms, rtol=0.0001, atol=0.0001)
    else:
        t_X = normalize(X, axis=0, norm=norm, return_norm=return_norm)
    assert type(t_X) == type(X)

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("norm", ['l1', 'l2', 'max'])
def test_sparse_normalize(sparse_clf_dataset, norm):  # noqa: F811
    X_np, X = sparse_clf_dataset

    def iscsc(X):
        return isinstance(X, cpu_sp.csc_matrix) or\
               isinstance(X, gpu_sp.csc_matrix)

    if iscsc(X):
        axis = 0
    else:
        axis = 1

    if norm == 'l1':
        norms = np.abs(X_np).sum(axis=axis)
    elif norm == 'l2':
        norms = np.linalg.norm(X_np, ord=2, axis=axis)
    elif norm == 'max':
        norms = np.max(abs(X_np), axis=axis)

    t_X_np = np.array(X_np, copy=True)

    if iscsc(X):
        t_X_np /= norms
    else:
        t_X_np = t_X_np.T
        t_X_np /= norms
        t_X_np = t_X_np.T

    t_X = normalize(X, axis=axis, norm=norm)
    assert type(t_X) == type(X)

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("strategy", ["mean", "most_frequent", "constant"])
@pytest.mark.parametrize("missing_values", [0., 1.])
def test_imputer(int_dataset, strategy, missing_values):  # noqa: F811
    X_np, X = int_dataset
    fill_value = np.random.randint(10, size=1)[0]

    fill_value = np.random.randint(10, size=1)[0]

    imputer = SimpleImputer(copy=True, missing_values=missing_values,
                            strategy=strategy, fill_value=fill_value)
    t_X = imputer.fit_transform(X)
    assert type(t_X) == type(X)

    imputer = OriginalSimpleImputer(copy=True, missing_values=missing_values,
                                    strategy=strategy, fill_value=fill_value)
    t_X_np = imputer.fit_transform(X_np)

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("strategy", ["mean", "most_frequent", "constant"])
@pytest.mark.parametrize("missing_values", [np.nan, 1.])
def test_imputer_sparse(sparse_int_dataset, strategy,  # noqa: F811
                        missing_values):
    X_np, X = sparse_int_dataset
    if isinstance(X, (cpu_sp.csr_matrix, gpu_sp.csr_matrix)):
        pytest.skip("unsupported sparse matrix")

    if X.format == 'csr':
        X_np = cpu_sp.csr_matrix(X_np)
    else:
        X_np = cpu_sp.csc_matrix(X_np)

    if np.isnan(missing_values):
        # Adding nan when missing value is nan
        random_loc = np.random.choice(X.nnz,
                                      int(X.nnz * 0.1),
                                      replace=False)
        X_np.data[random_loc] = np.nan
        X = X.copy()
        X.data[random_loc] = np.nan

    fill_value = np.random.randint(10, size=1)[0]

    imputer = SimpleImputer(copy=True, missing_values=missing_values,
                            strategy=strategy, fill_value=fill_value)
    t_X = imputer.fit_transform(X)
    assert type(t_X) == type(X)

    imputer = OriginalSimpleImputer(copy=True, missing_values=missing_values,
                                    strategy=strategy, fill_value=fill_value)
    t_X_np = imputer.fit_transform(X_np)
    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("order", ['C', 'F'])
def test_poly_features(clf_dataset, degree,  # noqa: F811
                       interaction_only, include_bias, order):
    X_np, X = clf_dataset

    polyfeatures = PolynomialFeatures(degree=degree, order=order,
                                      interaction_only=interaction_only,
                                      include_bias=include_bias)
    t_X = polyfeatures.fit_transform(X)
    assert type(X) == type(t_X)

    if isinstance(t_X, np.ndarray):
        if order == 'C':
            assert t_X.flags['C_CONTIGUOUS']
        elif order == 'F':
            assert t_X.flags['F_CONTIGUOUS']

    polyfeatures = OriginalPolFeat(degree=degree, order=order,
                                   interaction_only=interaction_only,
                                   include_bias=include_bias)
    X_sp = cpu_sp.csr_matrix(X_np)
    orig_t_X = polyfeatures.fit_transform(X_sp)

    assert_allclose(t_X, orig_t_X, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
def test_poly_features_sparse(sparse_clf_dataset, degree,  # noqa: F811
                              interaction_only, include_bias, order):
    X_np, X = sparse_clf_dataset

    polyfeatures = PolynomialFeatures(degree=degree, order=order,
                                      interaction_only=interaction_only,
                                      include_bias=include_bias)
    t_X = polyfeatures.fit_transform(X)
    assert type(t_X) == type(X)

    polyfeatures = OriginalPolFeat(degree=degree, order=order,
                                   interaction_only=interaction_only,
                                   include_bias=include_bias)
    X_sp = cpu_sp.csr_matrix(X_np)
    orig_t_X = polyfeatures.fit_transform(X_sp)

    assert_allclose(t_X, orig_t_X, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("value", [1.0, 42])
def test_add_dummy_feature(clf_dataset, value):  # noqa: F811
    X_np, X = clf_dataset

    t_X = add_dummy_feature(X, value=value)
    assert type(t_X) == type(X)

    t_X_np = original_adf(X_np, value=value)
    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("value", [1.0, 42])
def test_add_dummy_feature_sparse(sparse_clf_dataset, value):  # noqa: F811
    X_np, X = sparse_clf_dataset

    t_X = add_dummy_feature(X, value=value)
    assert type(t_X) == type(X)

    t_X_np = original_adf(X_np, value=value)
    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("threshold", [0., 1.])
def test_binarize(clf_dataset, threshold):  # noqa: F811
    X_np, X = clf_dataset

    t_X = binarize(X, threshold=threshold, copy=True)
    assert type(t_X) == type(X)

    t_X_np = np.array(X_np, copy=True)
    cond = X_np > threshold
    t_X_np[~cond] = 0
    t_X_np[cond] = 1

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("threshold", [0., 1.])
def test_binarize_sparse(sparse_clf_dataset, threshold):  # noqa: F811
    X_np, X = sparse_clf_dataset

    t_X = binarize(X, threshold=threshold, copy=True)
    assert type(t_X) == type(X)

    t_X_np = np.array(X_np, copy=True)
    cond = X_np > threshold
    t_X_np[~cond] = 0
    t_X_np[cond] = 1

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("threshold", [0., 1.])
def test_binarizer(clf_dataset, threshold):  # noqa: F811
    X_np, X = clf_dataset

    binarizer = Binarizer(threshold=threshold, copy=True)
    t_X = binarizer.fit_transform(X)
    assert type(t_X) == type(X)

    t_X_np = np.array(X_np, copy=True)
    cond = X_np > threshold
    t_X_np[~cond] = 0
    t_X_np[cond] = 1

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("threshold", [0., 1.])
def test_binarizer_sparse(sparse_clf_dataset, threshold):  # noqa: F811
    X_np, X = sparse_clf_dataset

    binarizer = Binarizer(threshold=threshold, copy=True)
    t_X = binarizer.fit_transform(X)
    assert type(t_X) == type(X)

    t_X_np = np.array(X_np, copy=True)
    cond = X_np > threshold
    t_X_np[~cond] = 0
    t_X_np[cond] = 1

    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)
