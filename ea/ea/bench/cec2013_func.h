#ifndef _CEC2013_FUNC_H 
#define _CEC2013_FUNC_H 1

extern double *OShift,*M,*y,*z,*x_bound;
extern int ini_flag,n_flag,func_flag;

void test_func(double *x, double *f, int nx, int mx,int func_num);

#endif
