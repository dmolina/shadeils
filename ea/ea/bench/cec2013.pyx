import cython
from ccec2013 cimport test_func
from libc.stdlib cimport malloc, free

cdef extern double *OShift
cdef extern double *M
cdef extern double *y
cdef extern double *z
cdef extern double *x_bound
cdef extern int ini_flag
cdef int fun_id

def cec2013_test_func(double[::1]x):
    cdef int dim
    cdef double fitness
    cdef double *sol

    dim = x.shape[0]

    sol = <double *> malloc(dim*cython.sizeof(double))

    if sol is NULL:
        raise MemoryError()

    for i in xrange(dim):
        sol[i] = x[i]

    test_func(sol, &fitness, dim, 1, fun_id)
    free(sol)
    return fitness

cdef class Benchmark:
    def __cinit__(self):
        global ini_flag
        global OShift, M, y, z, x_bound
        ini_flag = 0
        OShift = NULL
        M = NULL
        y = NULL
        z = NULL
        x_bound = NULL

    cpdef get_info(self, int fun, int dim):
        """ 
        Return the lower bound of the function
        """
        cdef double fun_best

        if (fun < 15):
            optimum = -1400+100*(fun-1)
        else:
            optimum = 100*(fun-14)

        return {'lower': -100, 'upper': 100, 'threshold': 0, 'best': optimum}

    def __dealloc(self):
        global OShift, M, y, z, x_bound

        if (OShift != NULL):
            free(OShift)

        if (M != NULL):
            free(M)

        if (y != NULL):
            free(y)

        if (z != NULL):
            free(z)

        if (x_bound != NULL):
            free(x_bound)

    cpdef get_function(self, int fun):
        """
        Evaluate the solution
        """
        global fun_id
        fun_id = fun
        return cec2013_test_func
