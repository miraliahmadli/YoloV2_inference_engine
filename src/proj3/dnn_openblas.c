#include <stdio.h>
#include <cblas.h>

void print_mat(const char *name, int r, int c, float *m)
{
    printf("%s =\n", name);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%.2lf ", m[i * c + j]);
        }
        printf("\n");
    }
}
void conv2d(double *C, double *A, double *B, int M, int K, int N){
    // M x K, K x N -> M x N (single precision, 's'gemm)
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        1,
        A, K,
        B, N,
        0,
        C, N
    );
}