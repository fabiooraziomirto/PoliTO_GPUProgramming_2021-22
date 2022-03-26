#include <stdio.h>
__global__ void kernFunction(int *a, int *b, int *c, int N){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= N) return;

	c[i] = a[i] + b[i];
	
}

#define N 100

int main(void){
	int threads = 1024;
	int blocks = N/threads + 1;
	
	// create pointers
	int *a, *b, *c;
	int *dev_a, *dev_b, *dev_c;

        // allocate memory on the CPU
	a = (int *) malloc (N * sizeof(int));
	b = (int *) malloc (N * sizeof(int));
	c = (int *) malloc (N * sizeof(int));

	// initialize the vector on the CPU
	for(int i = 0; i < N; i++){
		a[i] = i+2;
		b[i] = i+1;
	}

	for(int i = 0; i < N; i++){
		printf("a[%d] = %d\n", i, a[i]);
		printf("b[%d] = %d\n", i, b[i]);
	}

	// allocate memory on the GPU
	cudaMalloc( (void **) &dev_a, N * sizeof(int) );
	cudaMalloc( (void **) &dev_b, N * sizeof(int) );
	cudaMalloc( (void **) &dev_c, N * sizeof(int) );

	// copy the array 'a' and 'b' to the GPU
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	kernFunction<<<blocks, threads>>>(dev_a, dev_b, dev_c, N);

	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyHostToDevice);

	for(int i = 0; i < N; i++){
		if((a[i] + b[i] - c[i]) == 0)
			printf("Test Failed\n");
		else
			printf("Test Passed\n");
	}
	
	

	free(a);
	free(b);
	free(c);
	
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

 	return 0;	
}
