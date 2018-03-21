import cython
import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_blas as blas
cimport libc.math as math

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

#create loss_value_table for logistic loss
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def get_loss_value_table(int min_score, int max_score):

    cdef:
        int lookup_offset = -min_score
        np.ndarray[DTYPE_t, ndim=1, mode = "fortran"] loss_value_table = np.empty(max_score - min_score + 1, dtype = DTYPE)
        Py_ssize_t i = 0
        int s = min_score

    while (s < 0):
        loss_value_table[i] = math.log(1.0 + math.exp(s)) - s
        i += 1
        s += 1

    if s == 0:
        loss_value_table[i] = math.M_LN2
        i += 1
        s += 1

    while s <= max_score:
        loss_value_table[i] = math.log1p(math.exp(-s))
        i += 1
        s += 1
    return loss_value_table, lookup_offset

#create prob_value_table for logistic loss
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def get_prob_value_table(int min_score, int max_score):

    cdef:
        int lookup_offset = -min_score
        np.ndarray[DTYPE_t, ndim=1, mode = "fortran"] prob_value_table = np.empty(max_score - min_score + 1, dtype = DTYPE)
        Py_ssize_t i = 0
        DTYPE_t exp_value
        int s = min_score

    while (s < 0):
        exp_value = math.exp(s)
        prob_value_table[i] = (exp_value / (1.0 + exp_value)) - 1.0
        i += 1
        s += 1

    if (s == 0):
        prob_value_table[i] = -0.5
        i += 1
        s += 1

    while (s <= max_score):
        exp_value = math.exp(-s)
        prob_value_table[i] = (1.0 / (1.0 + exp_value)) - 1.0
        i += 1
        s += 1

    return prob_value_table, lookup_offset

#create both loss and prob tables for logistic loss
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def get_loss_value_and_prob_tables(int min_score, int max_score):

    cdef:
        int lookup_offset = -min_score
        int table_size = max_score - min_score + 1
        np.ndarray[DTYPE_t, ndim=1, mode = "fortran"] loss_value_table = np.empty(table_size, dtype = DTYPE)
        np.ndarray[DTYPE_t, ndim=1, mode = "fortran"] prob_value_table = np.empty(table_size, dtype = DTYPE)
        Py_ssize_t i = 0
        DTYPE_t exp_value
        int s = min_score

    while (s < 0):
        exp_value = math.exp(s)
        loss_value_table[i] = math.log(1.0 + exp_value) - s
        prob_value_table[i] = (exp_value / (1.0 + exp_value)) - 1.0
        i += 1
        s += 1

    if (s == 0):
        loss_value_table[i] = math.M_LN2
        prob_value_table[i] = -0.5
        i += 1
        s += 1

    while (s <= max_score):
        exp_value = math.exp(-s)
        loss_value_table[i] = math.log1p(exp_value)
        prob_value_table[i] = (1.0 / (1.0 + exp_value)) - 1.0
        i += 1
        s += 1

    return loss_value_table, prob_value_table, lookup_offset

##############################################################################################################
##############################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def log_loss_value(np.ndarray[DTYPE_t, ndim=2, mode="fortran"] Z,
np.ndarray[DTYPE_t, ndim=1, mode="fortran"] rho,
np.ndarray[DTYPE_t, ndim=1, mode="fortran"] loss_value_table,
int lookup_offset):

    cdef:
        int N = Z.shape[0]
        int D = Z.shape[1]
        int incx = 1 #increments of rho
        int incy = 1 #increments of y
        double alpha = 1.0
        double beta = 0.0
        np.ndarray[DTYPE_t, ndim=1, mode = "fortran"] y = np.empty(N, dtype = DTYPE)
        Py_ssize_t i
        DTYPE_t total_loss = 0.0

    #get scores using dgemv, which computes: y <- alpha * trans(Z) + beta * y
    #see also: (http://www.nag.com/numeric/fl/nagdoc_fl22/xhtml/F06/f06paf.xml)
    blas.dgemv("N", &N, &D, &alpha, &Z[0,0], &N, &rho[0], &incx, &beta, &y[0], &incy)

    #compute loss
    for i in range(N):
        total_loss += loss_value_table[(<int>y[i]) + lookup_offset]

    return total_loss/N

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def log_loss_value_from_scores(
    np.ndarray[DTYPE_t, ndim=1, mode="fortran"] scores,
    np.ndarray[DTYPE_t, ndim=1, mode="fortran"] loss_value_table,
    int lookup_offset):

    cdef:
        Py_ssize_t i
        Py_ssize_t N = scores.shape[0]
        DTYPE_t total_loss = 0.0

    #compute loss
    for i in range(N):
        total_loss += loss_value_table[((<int>scores[i]) + lookup_offset)]

    return total_loss/N

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def log_loss_value_and_slope(
            np.ndarray[DTYPE_t, ndim=2, mode="fortran"] Z,
            np.ndarray[DTYPE_t, ndim=1, mode="fortran"] rho,
            np.ndarray[DTYPE_t, ndim=1, mode="fortran"] loss_value_table,
            np.ndarray[DTYPE_t, ndim=1, mode="fortran"] prob_value_table,
            int lookup_offset):

    cdef:
        int N = Z.shape[0]
        int D = Z.shape[1]
        int lda = N
        int incx = 1 #increments of rho
        int incy = 1 #increments of y
        double alpha = 1.0
        double beta = 0.0
        Py_ssize_t i
        int lookup_index
        DTYPE_t total_loss = 0.0
        np.ndarray[DTYPE_t, ndim=1, mode = "fortran"] y = np.empty(N, dtype = DTYPE)
        np.ndarray[DTYPE_t, ndim=1, mode = "fortran"] loss_slope = np.empty(D, dtype = DTYPE)

    #get scores using dgemv, which computes: y <- alpha * trans(Z) + beta * y
    #see also: (http://www.nag.com/numeric/fl/nagdoc_fl22/xhtml/F06/f06paf.xml)
    blas.dgemv("N", &N, &D, &alpha, &Z[0,0], &lda, &rho[0], &incx, &beta, &y[0], &incy)

    #exponentiate scores, compute mean scores and probabilities
    for i in range(N):
        lookup_index = (<int> y[i]) + lookup_offset
        total_loss += loss_value_table[lookup_index]
        y[i] = prob_value_table[lookup_index]

    #compute loss slope
    alpha = 1.0/N
    blas.dgemv("T", &N, &D, &alpha, &Z[0,0], &lda, &y[0], &incx, &beta, &loss_slope[0], &incy)

    return (total_loss/N), loss_slope
