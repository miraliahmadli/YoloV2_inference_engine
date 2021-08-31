#include <cstdio>
# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
#include <iostream>
#include <algorithm>
#include "assert.h"
using namespace std;
 
# define IDX2C(i,j,ld) ((( j )*( ld ))+( i ))

 /*

    Convolution

*/

extern "C"{
    void conv2d(double *c, double *a, double *b, int m, int k, int n)
    {
        cudaError_t cudaStat; 
        cublasStatus_t stat; 
        cublasHandle_t handle; 
        double * d_a; // d_a - a on the device
        double * d_b; // d_b - b on the device
        double * d_c; // d_c - c on the device

        stat = cublasCreate (&handle ); // initialize CUBLAS context

        cudaStat = cudaMalloc ((void **) &d_a ,m*k*sizeof(double)); // device
        // memory alloc for a
        cudaStat = cudaMalloc ((void **) &d_b ,k*n*sizeof(double)); // device
        // memory alloc for b
        cudaStat = cudaMalloc ((void **) &d_c ,m*n*sizeof(double)); // device
        // memory alloc for c

        // copy matrices from the host to the device
        stat = cublasSetMatrix (m, k, sizeof (double), a, m, d_a ,m); //a -> d_a
        stat = cublasSetMatrix (k, n, sizeof (double), b, k, d_b ,k); //b -> d_b
        stat = cublasSetMatrix (m, n, sizeof (double), c, m, d_c ,m); //c -> d_c

        double al = 1.0; // al =1
        double bet = 0.0; // bet =1
        // matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
        // d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
        // al ,bet -scalars
        // cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE); 
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        n, m, k, 
                        &al, d_b, n, d_a, k, 
                        &bet, d_c, n);
        stat = cublasGetMatrix(m, n, sizeof(double) ,d_c ,m, c, m); // cp d_c - >c

        cudaFree (d_a ); // free device memory
        cudaFree (d_b ); // free device memory
        cudaFree (d_c ); // free device memory
        cublasDestroy (handle);
    }
}
 