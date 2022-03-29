#include <stdio.h>

#define NL 2048
#define ML 8



__global__ void convolution_1(float *M, float *P, float *N, int Mask_Width, int vector_Width){

	int tid= blockIdx.x * blockDim.x + threadIdx.x;
	float Pvalue = 0;
	int N_start_point = tid - (Mask_Width/2);

	for (int j = 0; j < Mask_Width; j++){
		if (N_start_point + j >= 0 && N_start_point+ j < vector_Width){
			Pvalue += N[N_start_point+ j] * M[j];
		}
	}
	P[tid] = Pvalue;
}

int main(void) {
	
	int j;
	float *M, *P, *N;
	float *dev_M, *dev_P,*dev_N;
	
	M = (float *) malloc(ML * sizeof(float));
	P = (float *) malloc(NL * sizeof(float));
	N = (float *) malloc(NL * sizeof(float));

	cudaMalloc(&dev_M, ML * sizeof(float));
	cudaMalloc(&dev_P, NL * sizeof(float));
	cudaMalloc(&dev_N, NL * sizeof(float));

	for (j = 0; j < NL; j++) {
		N[j] = j;
	}

	M[0] = 3;
	M[1] = 4;
	M[2] = 5;
	M[3] = 6;
	M[4] = 6;
	M[5] = 5;
	M[6] = 4;
	M[7] = 3;

	// run version with static shared memory
	
	cudaMemcpy(dev_M, M, ML*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_N, N, NL*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_P, P, NL*sizeof(float), cudaMemcpyHostToDevice);

	convolution_1<<<16, 128>>>(dev_M, dev_P, dev_N, ML, NL);
	// work also 
	// convolution_1<<<8, 256>>>(dev_M, dev_P, dev_N, ML, NL);
	cudaMemcpy(M, dev_M, ML*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(N, dev_N, NL*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(P, dev_P, NL*sizeof(float), cudaMemcpyDeviceToHost);

	printf("P = ");
	for (j = 0; j < NL; j++) {
		printf("%f ", P[j]);
	}	

	printf("\n");

	free(M);
	free(P);
	free(N);

	cudaFree(dev_M);
	cudaFree(dev_P);
	cudaFree(dev_N);

	return 0;
}
