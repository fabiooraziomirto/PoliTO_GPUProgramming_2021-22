#include <stdio.h>

__global__ void staticCreate(double *d, int n) {
	__shared__ int i[1024];
	int t = threadIdx.x;
	i[t] = t;
	d[t] = i[t];
	__syncthreads();
	
}

__global__ void dynamicCreate(double *d, int n){
	extern __shared__ int i[];
	int t = threadIdx.x;
	i[t] = t;
	d[t] = i[t];
	__syncthreads();
}

int main(void) {

	const int n = 1024;
	double array[n];
	
	double *array_d;
	cudaMalloc(&array_d, n * sizeof(double));

	// run version with static shared memory
	cudaMemcpy(array_d, array, n*sizeof(double), cudaMemcpyHostToDevice);

	staticCreate<<<1,n>>>(array_d, n);

	cudaMemcpy(array, array_d, n*sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; i++) {
		printf("%f ", array[i]);
		array[i] = 0;
	}
	printf("\n");

	// run dynamic shared memory version
	cudaMemcpy(array_d, array, n*sizeof(double), cudaMemcpyHostToDevice);

	dynamicCreate<<<1,n,n*sizeof(double)>>>(array_d, n);

	cudaMemcpy(array, array_d, n*sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; i++) {
		printf("%f ", array[i]);
	}
	printf("\n");
	
	return 0;
}
