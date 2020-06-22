
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


import cupy as cp


def csr_mean_variance_axis0(X):
    X = X.tocsc()
    return csc_mean_variance_axis0(X)


def csc_mean_variance_axis0(X):
    n_features = array.shape[1]

    means = cp.zeros(n_features)
    variances = cp.zeros(n_features)
    counts_nan = cp.zeros(n_features)

    start = X.indptr[0]
    for i, end in enumerate(X.indptr[1:]):
        col = X.data[start:end]
        means[i] = col.mean()
        variances[i] = col.var()
        counts_nan[i] = X.nnz - np.count_nonzero(np.isnan(col))
        start = end
    return means, variances, counts_nan


def incr_mean_variance_axis0(X, last_mean, last_var, last_n):
    if isinstance(X, cp.sparse.csr_matrix):
        new_mean, new_var, counts_nan = csr_mean_variance_axis0(X)
    elif isinstance(X, cp.sparse.csc_matrix):
        new_mean, new_var, counts_nan = csc_mean_variance_axis0(X)
    
    new_n = np.diff(X.indptr) - counts_nan

    # First pass
    is_first_pass = True
    for i in range(n_features):
        if last_n[i] > 0:
            is_first_pass = False
            break
    if is_first_pass:
        return new_mean, new_var, new_n

    # Next passes
    for i in range(n_features):
        if new_n[i] > 0:
            updated_n[i] = last_n[i] + new_n[i]
            last_over_new_n[i] = dtype(last_n[i]) / dtype(new_n[i])
            # Unnormalized stats
            last_mean[i] *= last_n[i]
            last_var[i] *= last_n[i]
            new_mean[i] *= new_n[i]
            new_var[i] *= new_n[i]
            # Update stats
            updated_var[i] = (
                last_var[i] + new_var[i] +
                last_over_new_n[i] / updated_n[i] *
                (last_mean[i] / last_over_new_n[i] - new_mean[i])**2
            )
            updated_mean[i] = (last_mean[i] + new_mean[i]) / updated_n[i]
            updated_var[i] /= updated_n[i]
        else:
            updated_var[i] = last_var[i]
            updated_mean[i] = last_mean[i]
            updated_n[i] = last_n[i]

    return updated_mean, updated_var, updated_n


def inplace_csr_row_normalize_l1(X):
    start = X.indptr[0]
    for end in X.indptr[1:]:
        col = X.data[start:end]
        col = abs(col)
        sum_ = col.sum()
        X.data[start:end] /= sum_
        start = end


def inplace_csr_row_normalize_l2(X):
    start = X.indptr[0]
    for end in X.indptr[1:]:
        col = X.data[start:end]
        col = cp.square(col)
        sum_ = col.sum()
        X.data[start:end] /= cp.sqrt(sum_)
        start = end
