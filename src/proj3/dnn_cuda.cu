#include <iostream>
#include <algorithm>
#include <stdio.h>
using namespace std;

#define BLOCK_SIZE 16
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString( err ), file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void add(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
    }

void print_mat(const char * name, int r, int c, double *m){
    printf("Printing %s\n", name);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%.2lf ", m[i * c + j]);
        }
        printf("\n");
    }
}

/*

    Leaky RELU

*/
__global__  void gpu_l_relu(double *res, int n, int k, int channel){
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockIdx.z * n * k;
    if( col < k && row < n) 
    {
        int index = offset + row * k + col;
        if(res[index] < 0.0) res[index] *= 0.1;
    }
}

extern "C" {
    void leaky_relu(double *res, int n, int k, int channel){
        // Allocate memory space on the device 
        double *dev_res;
        cudaMalloc((void **) &dev_res, sizeof(double)*n*k*channel);

        // copy matrix A and B from host to device memory
        cudaMemcpy(dev_res, res, sizeof(double)*n*k*channel, cudaMemcpyHostToDevice);

        unsigned int gridev_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int gridev_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int channels = channel;
        dim3 dimGrid(gridev_cols, gridev_rows, channels);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
        // Launch kernel 
        gpu_l_relu<<<dimGrid, dimBlock>>>(dev_res, n, k, channel);    

        // Transefr results from device to host 
        cudaMemcpy(res, dev_res, sizeof(double)*n*k*channel, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // free memory
        cudaFree(dev_res);
    }
}


/*

    Batch Norm

*/
__global__ void gpu_b_norm (double *res, double *mean, double *gamma, 
                    double *variance, int n, int k, int channel){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockIdx.z * n * k;
    if( col < k && row < n) {
            int index = offset + row * k + col;
            // double divisor = sqrt(variance[blockIdx.z] + epsilon);
            double divident = gamma[blockIdx.z] * (res[index] - mean[blockIdx.z]);
            res[index] = divident / variance[blockIdx.z];
    }
}

extern "C" {
    void batch_norm(double *res, double *mean, double *gamma, 
                    double *variance, int n, int k, int channel){
        // Allocate memory space on the device 
        double *dev_res, *dev_mean, *dev_gamma, *dev_variance;
        cudaMalloc((void **) &dev_res, sizeof(double)*n*k*channel);

        cudaMalloc((void **) &dev_mean, sizeof(double)*channel);
        cudaMalloc((void **) &dev_gamma, sizeof(double)*channel);
        cudaMalloc((void **) &dev_variance, sizeof(double)*channel);

        // copy matrix A and B from host to device memory
        cudaMemcpy(dev_res, res, sizeof(double)*n*k*channel, cudaMemcpyHostToDevice);

        cudaMemcpy(dev_mean, mean, sizeof(double)*channel, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_gamma, gamma, sizeof(double)*channel, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_variance, variance, sizeof(double)*channel, cudaMemcpyHostToDevice);

        unsigned int gridev_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int gridev_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int channels = channel;
        dim3 dimGrid(gridev_cols, gridev_rows, channels);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
        // Launch kernel 
        gpu_b_norm<<<dimGrid, dimBlock>>>(dev_res, dev_mean, dev_gamma, dev_variance, 
                                            n, k, channel);    

        // Transefr results from device to host 
        cudaMemcpy(res, dev_res, sizeof(double)*n*k*channel, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // free memory
        cudaFree(dev_res);
    }
}

/*

    Add Bias

*/

__global__ void gpu_add_bias (double *res, double *bias, int n, int k, int channel){
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockIdx.z * n * k;
    if( col < k && row < n) 
    {
        res[offset + row * k + col] += bias[blockIdx.z];
    }
}

extern "C" {
    void add_bias(double * C, double * bias, int n, int k, int channel){
        // Allocate memory space on the device 
        double *dev_b, *dev_c;
        cudaMalloc((void **) &dev_b, sizeof(double)*channel);
        cudaMalloc((void **) &dev_c, sizeof(double)*n*k*channel);

        // copy matrix A and B from host to device memory
        cudaMemcpy(dev_b, bias, sizeof(double)*channel, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_c, C, sizeof(double)*n*k*channel, cudaMemcpyHostToDevice);

        unsigned int gridev_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int gridev_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int channels = channel;
        dim3 dimGrid(gridev_cols, gridev_rows, channels);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
        // Launch kernel 
        gpu_add_bias<<<dimGrid, dimBlock>>>(dev_c, dev_b, n, k, channel);    

        // Transefr results from device to host 
        cudaMemcpy(C, dev_c, sizeof(double)*n*k*channel, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // free memory
        cudaFree(dev_b);
        cudaFree(dev_c);
    }
}

/*

    MAX Pool

*/

__global__ void gpu_max (double *res, double *cols, int n, int size){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n)
    {
        int offset = index;
        // printf("%d\n", offset);
        double max = -10000.0;
        for(int i =0; i<size; i++){
            if(cols[offset + i*n] > max){
                max = cols[offset + i*n];
            } 
        }
        // printf("Max value for index %d: %.2lf\n", index, max);
        res[index] = max;
    }
}

extern "C" {
    void maxpool(double * C, double * cols, int n, int size){
        // Allocate memory space on the device 
        double *dev_cols, *dev_c;

        cudaMalloc((void **) &dev_cols, sizeof(double)*n*size);
        cudaMalloc((void **) &dev_c, sizeof(double)*n);

        // copy matrix A and B from host to device memory
        cudaMemcpy(dev_cols, cols, sizeof(double)*n*size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_c, C, sizeof(double)*n, cudaMemcpyHostToDevice);

        // Launch kernel 
        gpu_max<<<(n + 1024 - 1)/1024, 1024>>>( dev_c, dev_cols, n, size);   

        // Transefr results from device to host 
        cudaMemcpy(C, dev_c, sizeof(double)*n, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // free memory
        cudaFree(dev_cols);
        cudaFree(dev_c);
    }
}

/*

    Convolution

*/
__global__ void gpu_multABtoC(double *a,double *b, double *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

extern "C"{
    void conv2d(double *C, double *A, double *B, int m, int n, int k)
    {
        // Allocate memory space on the device 
        double *dev_a, *dev_b, *dev_c;
        cudaMalloc((void **) &dev_a, sizeof(double)*m*n);
        cudaMalloc((void **) &dev_b, sizeof(double)*n*k);
        cudaMalloc((void **) &dev_c, sizeof(double)*m*k);

        // copy matrix A and B from host to device memory
        cudaMemcpy(dev_a, A, sizeof(double)*m*n, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, B, sizeof(double)*n*k, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_c, C, sizeof(double)*m*k, cudaMemcpyHostToDevice);

        unsigned int gridev_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int gridev_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(gridev_cols, gridev_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
        // Launch kernel 
        gpu_multABtoC<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, m, n, k);    

        // Transefr results from device to host 
        cudaMemcpy(C, dev_c, sizeof(double)*m*k, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // free memory
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
    }
}