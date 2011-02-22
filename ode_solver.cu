#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>

#include "kernel_runge_kutta.cu"

#define BLOCK_SIZE 32

float* random_init_y(const unsigned int size) {
	float* to_return = (float *) malloc(size * sizeof(float));
	for(int i=0; i<size; i++) {
		to_return[i] = rand();
	}
	return to_return;
}

float* increasing_init_y(const unsigned int size) {
	float* to_return = (float *) malloc(size * sizeof(float));
	for(int i=0; i<size; i++) {
		to_return[i] = i;
	}
	return to_return;
}

void solve_ode_on_gpu(
	float f(float t, float y),
	const float init_t,
	const float* init_y,
	const unsigned int size,
	const float h,
	const unsigned int maximum_steps,
	const float target_time,
	float* result,
	unsigned int* last_step
) {

	int grid_size = ceil(sqrt(size / (BLOCK_SIZE * BLOCK_SIZE))) + 1;
	dim3 grid_dim(grid_size, grid_size);
	dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
	printf("\t[INFO] executing kernel (%i, %i) (%i, %i)\n", grid_size, grid_size, BLOCK_SIZE, BLOCK_SIZE);
	solve_ode<<<grid_dim, block_dim>>>(
		f,
		init_t,
		init_y,
		size,
		h,
		maximum_steps,
		target_time,
		result,
		last_step
	);
	cudaThreadSynchronize();
	cudaError error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "\t[ERROR] execution failed: %s\n", cudaGetErrorString(error));
		exit(1);
	}
}

int main(int argc, char **argv) {

	cudaError error = cudaSuccess;

	// Set size
	if (argc < 2) {
		fprintf(stderr, "\t[ERROR] you must specify size of test \n");
		exit(1);
	}
	unsigned int size = atoi(argv[1]);
	printf("\t[INFO] setting size of init vector to %i\n", size);

	// Set maximum number of steps
	if (argc < 3) {
		fprintf(stderr, "\t[ERROR] you must specify maximum number of simulation steps \n");
		exit(1);
	}
	unsigned int maximum_steps = atoi(argv[2]);
	printf("\t[INFO] setting maximum number of simulation steps to %i\n", maximum_steps);

	// Set step size
	if (argc < 4) {
		fprintf(stderr, "\t[ERROR] you must specify size of the time step \n");
		exit(1);
	}
	float step_size = atof(argv[3]);
	printf("\t[INFO] setting size of the time stepto %i\n", step_size);

	// Set target time
	if (argc < 5) {
		fprintf(stderr, "\t[ERROR] you must specify target time \n");
		exit(1);
	}
	float target_time = atof(argv[4]);
	printf("\t[INFO] setting target time to %f\n", target_time);	

	// Set device
	int device = 0;
	if (argc == 6)  {
		device = atoi(argv[5]);
	}
	printf("\t[INFO] using device %i\n", device);
	error = cudaSetDevice(device);
	if (error != cudaSuccess) {
		fprintf(stderr, "\t[ERROR] cannot set CUDA device (%i): %s\n", device, cudaGetErrorString(error));
		exit(1);
	}
	
	// Prepare data
	printf("\t[INFO] generating random init vector\n");
	float* cpu_init_y = increasing_init_y(size);
	float* gpu_init_y = NULL;
	cudaMalloc((void**) &gpu_init_y, size * sizeof(float));
	printf("\t[INFO] copying init vector to GPU\n");
	cudaMemcpy(gpu_init_y, cpu_init_y, size* sizeof(float), cudaMemcpyHostToDevice);
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "\t[ERROR] copying init vector to GPU failed: %s\n", cudaGetErrorString(error));
		exit(1);
	}
	printf("\t[INFO] preparing memory for GPU computation\n");
	float* gpu_result = NULL;
	unsigned int* gpu_last_step = NULL;
	cudaMalloc((void**) &gpu_result, size * (ceil(target_time/step_size)) * sizeof(float));
	cudaMalloc((void**) &gpu_last_step, size * sizeof(unsigned int));
	
	// Execute simulation
	printf("\t[INFO] executing simulation on GPU\n");	
	solve_ode_on_gpu(
		example_f,
		0,
		gpu_init_y,
		size,
		step_size,
		maximum_steps,
		target_time,
		gpu_result,
		gpu_last_step
	);

	// Retrieve data from GPU
	printf("\t[INFO] retrieving data from GPU\n");
	float* cpu_result = (float *) malloc(size * maximum_steps * sizeof(float));
	unsigned int* cpu_last_step = (unsigned int*) malloc(size * sizeof(unsigned int));
	cudaMemcpy(cpu_result, gpu_result, size * (ceil(target_time/step_size)) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_last_step, gpu_last_step, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "\t[ERROR] copying failed: %s\n", cudaGetErrorString(error));
		exit(1);
	}

/*	printf("\n");
	for(int thread=0; thread<size; thread++) {
		printf("\t[%i:%i]\t",thread, cpu_last_step[thread]); 
		for(int step=0; step<cpu_last_step[thread]; step++) {
			printf("%4.2f ", cpu_result[step * size + thread]);
		}
		printf("\n");
	}
	printf("\n");
*/
	// Clear data
	printf("\t[INFO] cleaning data\n");	
	free(cpu_init_y);
	free(cpu_result);
	free(cpu_last_step);
	cudaFree(gpu_init_y);
	cudaFree(gpu_result);
	cudaFree(gpu_last_step);	
}
