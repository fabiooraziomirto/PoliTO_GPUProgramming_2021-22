#include <stdio.h>

__global__ void mean_1(float *input, float *mean_output, int num_elem) {

	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int tid_idx;
	float d0, d1, d2;
	unsigned int tid_idx_max = (num_elem - 1);

	for (int i = 0; i < num_elem; i+=3) {
		tid_idx = (tid * 3);
		printf("tid %d\n", tid_idx);
		if (tid_idx < tid_idx_max) {
			d0 = input[tid_idx];
			printf("d0 = %f\n", d0);
			d1 = input[tid_idx+1];
			printf("d1 = %f\n", d1);
			d2 = input[tid_idx+2];
			printf("d2 = %f\n", d2);
			if((d0+d1+d2)/3 != 0)
				mean_output[tid] = (d0+d1+d2)/3;	
		}
	}
}

#define N 27

int main(void){
	int threads = 8;
	int blocks = N/threads + 1;
	
	// create pointers
	float *data, *mean;
	float *dev_data, *dev_mean;

        // allocate memory on the CPU
	data = (float *) malloc (N * sizeof(float));
	mean = (float *) malloc ((N/3) * sizeof(float));

	// initialize the vector on the CPU
	for(int i = 0; i < N; i++){
		data[i] = i+1.1*i;	
	}

	// allocate memory on the GPU
	cudaMalloc( (void **) &dev_data, N * sizeof(float) );
	cudaMalloc( (void **) &dev_mean, (N/3) * sizeof(float) );

	// copy the array 'a' and 'b' to the GPU
	cudaMemcpy(dev_data, data, N * sizeof(float), cudaMemcpyHostToDevice);

	mean_1<<<blocks, threads>>>(dev_data, dev_mean, N);
	
	if(cudaMemcpy(mean, dev_mean, (N/3) * sizeof(float), cudaMemcpyDeviceToHost) == 0)
		printf("SUCCESS\n");

	for(int i = 0; i < N/3; i++){
		printf("%f ", mean[i]);	
	}
	
	printf("\n");

	free(data);
	free(mean);
	
	cudaFree(dev_data);
	cudaFree(dev_mean);	
	
 	return 0;	
}
