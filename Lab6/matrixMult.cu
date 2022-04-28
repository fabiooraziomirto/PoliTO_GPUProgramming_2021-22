#include <stdio.h>


#define N 3
#define MASK_DIM 3

#define MASK_OFFSET (MASK_DIM/2)



__global__ void matrix_mul(float *m, float *n, float *result){

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	

	float temp = 0;

	if(row < N && col < N){
		for(int i = 0; i < N; i++){
			temp += m[row * N + i] * n[i * N + col];
		}
	}
	
	result[row * N + col] = temp;
}

void generate_matrix(float *matrix, int dim){
	for(int i = 0; i < dim; i++){
		for(int j = 0; j < dim; j++){
			matrix[i * dim + j] = rand() % 9;
		}
	}
}



int main(void) {
	
	float *m;
	float *result;
	float *n;

	m = (float *) malloc(N * N  * sizeof(float));
	
	result = (float *) malloc(N * N  * sizeof(float));

	n = (float *) malloc(MASK_DIM * MASK_DIM  * sizeof(float));

	generate_matrix(m, N);
	generate_matrix(n, N);

	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			printf("%f ", m[i * N +j]);
		}
		printf("\n");
	}

	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			printf("%f ", n[i * N +j]);
		}
		printf("\n");
	}


	
	float *m_dev, *n_dev, *result_dev;
	
	
	cudaMalloc(&m_dev, N * N * sizeof(float));
	cudaMalloc(&n_dev, MASK_DIM * MASK_DIM * sizeof(float));
	cudaMalloc(&result_dev, N*N*sizeof(float));
	

	cudaMemcpy(m_dev, m, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(n_dev, n, MASK_DIM * MASK_DIM *sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(result_dev, result, N*N*sizeof(float), cudaMemcpyHostToDevice);
	
	int THREADS = N;
	int BLOCKS = (N + THREADS - 1)/THREADS;

	dim3 gridDim(BLOCKS, BLOCKS);
	dim3 blockDim(THREADS, THREADS);

	matrix_mul<<<gridDim, blockDim>>>(m_dev, n_dev, result_dev);

	cudaMemcpy(m, m_dev, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(result, result_dev, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(n, n_dev, MASK_DIM * MASK_DIM *sizeof(float), cudaMemcpyDeviceToHost);
	

	printf("RESULT\n");
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			printf("%f ", result[i * N +j]);
		}
		printf("\n");
	}

	free(result);
	free(m);
	free(n);

	cudaFree(m_dev);
	cudaFree(n_dev);
	cudaFree(result_dev);


	return 0;
}
