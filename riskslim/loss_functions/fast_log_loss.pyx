import cython
import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_blas as blas
cimport libc.math as math

DTYPE = np.float64
ctypedef np.float64_t DTYPE_T

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def log_loss_value(np.ndarray[DTYPE_T, ndim=2, mode="fortran"] Z, np.ndarray[DTYPE_T, ndim=1, mode="fortran"] rho):

    cdef:
        int N = Z.shape[0]
        int D = Z.shape[1]
        int lda = N
        int incx = 1 #increments of rho
        int incy = 1 #increments of y
        double alpha = 1.0
        double beta = 0.0
        np.ndarray[DTYPE_T, ndim=1, mode = "fortran"] y = np.empty(N, dtype = DTYPE)
        Py_ssize_t i
        DTYPE_T total_loss = 0.0
        int zero_score_cnt = 0

    #compute scores
    #calls dgemv from BLZS which computes y = alpha * trans(Z) + beta * y
    #see: http://www.nag.com/numeric/fl/nagdoc_fl22/xhtml/F06/f06paf.xml
    blas.dgemv("N", &N, &D, &alpha, &Z[0,0], &lda, &rho[0], &incx, &beta, &y[0], &incy)

    #compute loss
    for i in range(N):
        if (y[i] < 0):
            total_loss += math.log(1.0 + math.exp(y[i])) - y[i]
        elif (y[i] > 0):
            total_loss += math.log1p(math.exp(-y[i]))
        else:
            zero_score_cnt += 1

    total_loss += zero_score_cnt * math.M_LN2
    return total_loss/N

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def log_loss_value_and_slope(np.ndarray[DTYPE_T, ndim=2, mode="fortran"] Z, np.ndarray[DTYPE_T, ndim=1, mode="fortran"] rho):

    cdef:
        int N = Z.shape[0]
        int D = Z.shape[1]
        int lda = N
        int incx = 1 #increments of rho
        int incy = 1 #increments of y
        double alpha = 1.0
        double beta = 0.0
        Py_ssize_t i
        DTYPE_T total_loss = 0.0
        DTYPE_T exp_value
        np.ndarray[DTYPE_T, ndim=1, mode = "fortran"] y = np.empty(N, dtype = DTYPE)
        np.ndarray[DTYPE_T, ndim=1, mode = "fortran"] loss_slope = np.empty(D, dtype = DTYPE)

    #compute scores
    #calls dgemv from BLAS which computes y = alpha * trans(Z) + beta * y
    #see: http://www.nag.com/numeric/fl/nagdoc_fl22/xhtml/F06/f06paf.xml
    blas.dgemv("N", &N, &D, &alpha, &Z[0,0], &lda, &rho[0], &incx, &beta, &y[0], &incy)

    #exponentiate scores, compute mean scores and probabilities
    for i in range(N):
        if y[i] < 0:
            exp_value = math.exp(y[i])
            total_loss += math.log(1.0 + exp_value) - y[i]
            y[i] = (exp_value / (1.0 + exp_value)) - 1.0
        else:
            exp_value = math.exp(-y[i])
            total_loss += math.log1p(exp_value)
            y[i] = (1.0 / (1.0 + exp_value)) - 1.0

    #compute loss slope
    alpha = 1.0/N
    blas.dgemv("T", &N, &D, &alpha, &Z[0,0], &lda, &y[0], &incx, &beta, &loss_slope[0], &incy)
    return (total_loss/N), loss_slope

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
def log_loss_value_from_scores(np.ndarray[DTYPE_T, ndim=1, mode="fortran"] scores):

    cdef:
        Py_ssize_t N = scores.shape[0]
        DTYPE_T total_loss = 0.0
        int zero_score_cnt = 0
        Py_ssize_t i
        DTYPE_T s

    #compute loss
    for i in range(N):
        s = scores[i]
        if s < 0:
            total_loss += math.log(1.0 + math.exp(s)) - s
        elif s > 0:
            total_loss += math.log1p(math.exp(-s))
        else:
            zero_score_cnt += 1

    total_loss += zero_score_cnt * math.M_LN2
    return total_loss/N
