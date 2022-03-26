#include <stdio.h>

__global__ void parallel_sort(int *data, int num_elem) {

	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int tid_idx;
	unsigned int offset = 0;
	unsigned int d0, d1;
	unsigned int tid_idx_max = (num_elem - 1);

	for (int i = 0; i < num_elem; i++) {
		tid_idx = (tid * 2) + offset;
		if (tid_idx < tid_idx_max) {
			d0 = data[tid_idx];
			d1 = data[tid_idx + 1];
			if (d0 > d1){
				data[tid_idx] = d1;
				data[tid_idx + 1] = d0;
			}
		}
		if (offset == 0)
			offset = 1;
		else
			offset = 0;
	}

	

}

#define N 100

int main(void){
	int threads = 1024;
	int blocks = N/threads + 1;
	
	// create pointers
	int *data;
	int *dev_data;

        // allocate memory on the CPU
	data = (int *) malloc (N * sizeof(int));

	// initialize the vector on the CPU
	for(int i = N; i > 0; i--){
		data[i] = i;
	}

	// allocate memory on the GPU
	cudaMalloc( (void **) &dev_data, N * sizeof(int) );

	// copy the array 'a' and 'b' to the GPU
	cudaMemcpy(dev_data, data, N * sizeof(int), cudaMemcpyHostToDevice);

	parallel_sort<<<blocks, threads>>>(dev_data, N);
	
	cudaMemcpy(data, dev_data, N * sizeof(int), cudaMemcpyHostToDevice);
	
	for(int i = 0; i < N; i++){
		printf("%d ", data[i]);
	}

	free(data);
	
	cudaFree(dev_data);

 	return 0;	
}
