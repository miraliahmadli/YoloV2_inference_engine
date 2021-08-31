#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>
#include<unistd.h> 
#include<stdlib.h> 
#define min(a,b)            (((a) < (b)) ? (a) : (b))
int BLOCK, NUM_THR;
double *A_p;
double *B_p;
int M, N, K;

void *maximum(void* arg)
{
    int *data = (int *)arg;
    double max = -1000000.0;
    int i = 0, j =0;

    int col = *data;
    double p[BLOCK];
    for (j = 0; j < BLOCK; j++){
        max = -1000000.0;
        for (i = 0; i < M; i++){
            if(A_p[col + j + i*N] > max) max = A_p[col + j + i*N];
        }
        p[j] = max;
    }
    free(arg);
    pthread_exit(p);
}

void maxpool(double* C, double *cols, int M_p, int N_p){
    M = M_p; N = N_p;
    A_p = (double *)malloc(M*N*sizeof(double));
    int size = N;
    BLOCK = 169;
    NUM_THR = size / BLOCK;
    int i, j;
  
    for(i = 0; i < M*N; i++) A_p[i] = cols[i];

    pthread_t *threads;
    threads = (pthread_t*)malloc(NUM_THR*sizeof(pthread_t));
    int counter = 0;
    for (j = 0; j < N; j = j + BLOCK){
        int *data;
        data = calloc(1,sizeof(int));
        *data = j;
        if(pthread_create(&threads[counter++], NULL, maximum, (void*)data) != 0){
            printf("Create failed at %d %d\n", i, j);
            exit(-1);
            return;
        }
    }
    double *res;
    for (i = 0; i < NUM_THR; i++) {
        void *k;
        pthread_join(threads[i], &k);
        res = k;
        j = 0;
        while(j < BLOCK){
            C[i * BLOCK + j] = res[j];
            j++;
        }
    }
    double max ;
    for(j = 0; j < N; j++){
        max = -1000.0;
        for(int i =0; i<M; i++){
            if(cols[j + i*N] > max){
                max = cols[j + i*N];
            } 
        }
        if(C[j] - max > 0.01){
            printf("FAILED at C[%d], %f is not %f", j, max, C[j]);
            exit(-1);
        }
    }
    
    free(A_p);
}

/*

    Convolution

*/
void *mult(void* arg)
{
    int *data = (int *)arg;
    double sum = 0.0;
    int i = 0, j =0;

    int row = *data;
    int col = *(data + 1);
    double p[BLOCK];
    __m256d col_4, row_4, res_4;
    for (j = 0; j < BLOCK; j++){
        sum = 0.0;
        for (i = 0; i < K; i = i + 4){
            col_4 = _mm256_loadu_pd(&A_p[row * K + i]);
            row_4 = _mm256_loadu_pd(&B_p[(col + j) * K + i]);
            res_4 = _mm256_mul_pd(col_4, row_4);
            res_4 = _mm256_hadd_pd(res_4, res_4);
            sum += (*(double *)&res_4[0]) + (*(double *)&res_4[2]);
        }
        p[j] = sum;
    }
    free(arg);
    pthread_exit(p);
}

void conv2d(double *C, double *A, double *B, int M_p, int K_p, int N_p){
    M = M_p; N = N_p; K = K_p;
    A_p = (double *)malloc(M*K*sizeof(double));
    B_p = (double *)malloc(N*K*sizeof(double));
    int size = M*N;
    if(size < 30000) BLOCK = 1;
    else BLOCK = 169;
    NUM_THR = size / BLOCK;

    int i, j;
  
    for(i = 0; i < M*K; i++) A_p[i] = A[i];
    for(i = 0; i < N*K; i++) B_p[i] = B[i];

    pthread_t *threads;
    threads = (pthread_t*)malloc(NUM_THR*sizeof(pthread_t));
    // printf("YUUUUP\n");
    int counter = 0;
    for (i = 0; i < M; i++){
        for (j = 0; j < N; j = j + BLOCK){
            int *data;
            data = calloc(2,sizeof(int));
            *data = i;
            *(data + 1) = j;
            if(pthread_create(&threads[counter++], NULL, mult, (void*)data) != 0){
                printf("Create failed at %d %d\n", i, j);
                exit(-1);
                return;
            }
        }
    }
    double *res;
    for (i = 0; i < NUM_THR; i++) {
        void *k;
        pthread_join(threads[i], &k);
        res = k;
        j = 0;
        while(j < BLOCK){
            C[i * BLOCK + j] = res[j];
            j++;
        }
    }
    
    free(A_p);
    free(B_p);
}