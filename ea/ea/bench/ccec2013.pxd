cdef extern from "cec2013_func.h":
    extern double *M
    extern double *y
    extern double *z
    extern double *x_bound
    extern int ini_flag
    void test_func(double *x, double *f, int nx, int mx,int func_num)
