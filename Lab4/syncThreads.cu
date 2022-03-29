#include <stdio.h>
#include <math.h>

#define N 3

__global__ void simple_kernel(float* x, float* y) {

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if ( (i % blockDim.x) == 0)
		x[i] = x[i] + 1;
	if ( (i % blockDim.x) == 1)
		y[i] = x[i-1] / 2.0;
	if ( (i % blockDim.x) == 2)
		x[i-1] = cos(y[i-1]);
	if ( (i % blockDim.x) == 3)
		y[i-1] = x[i-2] * y[i-2] + x[i-3];
	__syncthreads();

}

int main(void) {

	float *x, *y;
	float *dev_x, *dev_y;
	
	x = (float *) malloc(N * sizeof(float));
	y = (float *) malloc(N * sizeof(float));

	cudaMalloc(&dev_x, N * sizeof(float));
	cudaMalloc(&dev_y, N * sizeof(float));
	
	x[0] = 45;

	// run version with static shared memory
	cudaMemcpy(dev_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

	simple_kernel<<<1, 32>>>(dev_x, dev_y);

	cudaMemcpy(x, dev_x, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, dev_y, N*sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		printf("x = %f ", x[i]);
		printf("y = %f ", y[i]);
	}
	printf("\n");

	free(x);
	free(y);

	cudaFree(dev_x);
	cudaFree(dev_y);

	return 0;
}
