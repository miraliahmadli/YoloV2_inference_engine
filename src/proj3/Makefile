# # CUDA
# cuda_lib.so: dnn_cuda.cu
# 	nvcc -Xcompiler -fPIC -shared -o $@ $^

# # CuBLAS
# cublas_lib.so: dnn_cublas.cu
# 	nvcc -Xcompiler -fPIC -shared -o $@ $^ -lcublas

# # OpenBLAS
# openblas_lib.so: dnn_openblas.c
# 	gcc -shared -fPIC -o $@ $^ -lopenblas

# AVX
avx_lib.so: dnn_avx.c
	gcc -shared -fPIC -o $@ $^ -mavx2 -pthread

clean:
	rm -f cublas_lib.so cuda_lib.so openblas_lib.so avx_lib.so
