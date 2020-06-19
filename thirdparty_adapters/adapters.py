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

import numpy as np
import cupy as cp
from cuml.common import input_to_cuml_array
from cupy.sparse import csr_matrix as gpu_csr_matrix
from cupy.sparse import csc_matrix as gpu_csc_matrix
from scipy.sparse import csr_matrix as cpu_csr_matrix
from scipy import sparse as cpu_sparse
from cupy import sparse as gpu_sparse

from numpy import ndarray as numpyArray
from cupy import ndarray as cupyArray
from cudf.core import Series as cuSeries
from cudf.core import DataFrame as cuDataFrame
from pandas import Series as pdSeries
from pandas import DataFrame as pdDataFrame
from numba.cuda import devicearray as numbaArray


numeric_types = [
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.intp, np.uintp,
    np.float32, np.float64,
    np.complex64, np.complex128
]


def check_sparse(array, accept_sparse):
    if cpu_sparse.issparse(array) or gpu_sparse.issparse(array):
        if not accept_sparse:
            err_msg = "This algorithm does not support sparse matrices."
            raise ValueError(err_msg)


def check_f16(dtype):
    # fp16 is not supported, so remove from the list of dtypes if present
    if isinstance(dtype, (list, tuple)):
        return [d for d in dtype if d != np.float16]
    elif dtype == np.float16:
        raise NotImplementedError("Float16 not supported by cuML")


def check_dtype(dtype, dtypes):
    if dtype not in dtypes:
        raise ValueError("Wrong dtype")


def check_finite(X, force_all_finite):
    if force_all_finite is True and not cp.all(cp.isfinite(X)):
        raise ValueError("Non-finite value encountered in array")


def check_array(array, accept_sparse=False, accept_large_sparse=True,
                dtype=numeric_types, order=None, copy=False,
                force_all_finite=True, ensure_2d=True, allow_nd=False,
                ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=None, estimator=None):

    dtype = check_f16(dtype)

    check_sparse(array, accept_sparse)

    if cpu_sparse.issparse(array):
        check_dtype(array.dtype, dtype)
        if isinstance(array, cpu_csr_matrix):
            array = gpu_csr_matrix(array)
        else:
            array = gpu_csc_matrix(array)
        check_finite(array, force_all_finite)
        return array

    if gpu_sparse.issparse(array):
        check_dtype(array.dtype, dtype)
        check_finite(array, force_all_finite)
        if isinstance(array, gpu_csr_matrix):
            array = gpu_csr_matrix(array, copy=True)
        else:
            array = gpu_csc_matrix(array, copy=True)
        return array

    X, n_rows, n_cols, dtype = input_to_cuml_array(array,
                                                   deepcopy=copy,
                                                   check_dtype=dtype)

    X = X.to_output('cupy')

    check_finite(X, force_all_finite)

    return X


_input_type_to_str = {
    numpyArray: 'numpy',
    cupyArray: 'cupy',
    cuSeries: 'cudf',
    cuDataFrame: 'cudf',
    pdSeries: 'numpy',
    pdDataFrame: 'numpy'
}


def get_input_type(input):
    # function to access _input_to_str, while still using the correct
    # numba check for a numba device_array
    if type(input) in _input_type_to_str.keys():
        return _input_type_to_str[type(input)]
    elif numbaArray.is_cuda_ndarray(input):
        return 'numba'
    elif isinstance(input, cpu_sparse.csr_matrix):
        return 'numpy_csr'
    elif isinstance(input, cpu_sparse.csc_matrix):
        return 'numpy_csc'
    elif isinstance(input, gpu_sparse.csr_matrix):
        return 'cupy_csr'
    elif isinstance(input, gpu_sparse.csc_matrix):
        return 'cupy_csc'
    else:
        return 'cupy'


def to_output_type(array, output_type, order='F'):
    if output_type == 'numpy_csr':
        return cpu_sparse.csr_matrix(array.get())
    if output_type == 'numpy_csc':
        return cpu_sparse.csc_matrix(array.get())
    if output_type == 'cupy_csr':
        return array
    if output_type == 'cupy_csc':
        return array

    if cpu_sparse.issparse(array):
        if output_type == 'numpy':
            return array.todense()
        elif output_type == 'cupy':
            return cp.array(array.todense())
    elif gpu_sparse.issparse(array):
        if output_type == 'numpy':
            return cp.asnumpy(array.todense())
        elif output_type == 'cupy':
            return array.todense()

    cuml_array = input_to_cuml_array(array, order=order)[0]
    return cuml_array.to_output(output_type)
