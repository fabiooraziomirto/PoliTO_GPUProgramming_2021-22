#include <stdio.h>


#define N 7
#define MASK_DIM 7

#define MASK_OFFSET (MASK_DIM/2)



__global__ void convolution_2(float *matrix, float *mask, float *result){

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;


	int start_r = row - MASK_OFFSET;
	int start_c = col - MASK_OFFSET;

	float temp = 0;

	for(int i = 0; i < MASK_DIM; i++){
		for(int j = 0; j < MASK_DIM; j++){
			if((start_r + i) >= 0 && (start_r + i) < N){
				if((start_c + j) >= 0 && (start_c + j) < N){
					temp += matrix[(start_r+i) * N + start_c + j] * 
						mask[i * MASK_DIM + j];	
				}
			}
		}
	}
	
	result[row * N + col] = temp;
}

void generate_matrix(float *matrix, int dim){
	for(int i = 0; i < dim; i++){
		for(int j = 0; j < dim; j++){
			matrix[i * dim + j] = rand() % 20 + i*0.2;
		}
	}
}


void verify_result(float *m, float *mask, float *result){

	float temp;
	int off_r;
	int off_c;

	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			temp = 0;
		for(int k = 0; k < MASK_DIM; k++){
			off_r = i - MASK_OFFSET + k;
		for(int l = 0; l < N; l++){
			off_c = j - MASK_OFFSET + l;
		if(off_r >= 0 && off_r < N){
			if(off_c >= 0 && off_c < N){
				temp += m[off_r * N + off_c] * mask[k * MASK_DIM + l];
				}
			}
		}
	}
		result[i*N + j] = temp;
	}
	}
}


int main(void) {
	
	float *matrix;
	float *result;
	float *mask;

	matrix = (float *) malloc(N * N  * sizeof(float));
	
	result = (float *) malloc(N * N  * sizeof(float));

	mask = (float *) malloc(MASK_DIM * MASK_DIM  * sizeof(float));

	generate_matrix(matrix, N);
	generate_matrix(mask, MASK_DIM);

	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			printf("%f ", matrix[i * N +j]);
		}
		printf("\n");
	}


	
	float *matrix_dev, *mask_dev, *result_dev;
	
	
	cudaMalloc(&matrix_dev, N * N * sizeof(float));
	cudaMalloc(&mask_dev, MASK_DIM * MASK_DIM * sizeof(float));
	cudaMalloc(&result_dev, N*N*sizeof(float));
	

	cudaMemcpy(matrix_dev, matrix, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(mask_dev, mask, MASK_DIM * MASK_DIM *sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(result_dev, result, N*N*sizeof(float), cudaMemcpyHostToDevice);
	
	int THREADS = 8;
	int BLOCKS = (N + THREADS - 1)/THREADS;

	dim3 gridDim(BLOCKS, BLOCKS);
	dim3 blockDim(THREADS, THREADS);

	convolution_2<<<gridDim, blockDim>>>(matrix_dev, mask_dev, result_dev);

	cudaMemcpy(matrix, matrix_dev, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(result, result_dev, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(mask, mask_dev, MASK_DIM * MASK_DIM *sizeof(float), cudaMemcpyDeviceToHost);
	

	printf("RESULT\n");
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			printf("%f ", result[i * N +j]);
		}
		printf("\n");
	}

	verify_result(matrix, mask, result);

	printf("RESULT\n");
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			printf("%f ", result[i * N +j]);
		}
		printf("\n");
	}

	free(result);
	free(matrix);

	cudaFree(matrix_dev);
	cudaFree(mask_dev);
	cudaFree(result_dev);


	return 0;
}
