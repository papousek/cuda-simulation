#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include "num_simulation_kernels.h"

#define BLOCK_SIZE 32

int 	size;
int 	maximum_steps;
float 	step_size;
float 	target_time;
int 	vector_size;
int 	device = 0;

float* 	gpu_init_vectors;
float* 	gpu_result;
int* 	gpu_number_of_executed_steps;
int* 	gpu_return_code;

float*	cpu_init_vectors;
float*	cpu_result;
int*	cpu_number_of_executed_steps;
int*	cpu_return_code;

void print_message(char* type, char* content) {
	printf("\t[%s]\t%s\n", type, content);
}

void print_info(char* content) {
	print_message("INFO", content);
}

void clean() {
	print_info("cleaning memory");
	
	if (gpu_init_vectors) cudaFree(gpu_init_vectors);
	if (gpu_result) cudaFree(gpu_result);
	if (gpu_number_of_executed_steps) cudaFree(gpu_number_of_executed_steps);
	if (gpu_return_code) cudaFree(gpu_return_code);
	
	if (cpu_init_vectors) free(cpu_init_vectors);	
	if (cpu_number_of_executed_steps) free(cpu_number_of_executed_steps);
}

void print_error(char* content) {
	print_message("ERROR", content);
	clean();
	exit(1);
}

float* init_increasing_vectors(const unsigned int size) {
	float* to_return = (float *) malloc(size * sizeof(float));
	for(int i=0; i<size; i++) {
		to_return[i] = i+1;
	}
	return to_return;
}

void check_cuda_error(char* msg) {
	cudaThreadSynchronize();
	cudaError error = cudaGetLastError();
	if(error != cudaSuccess) {
		char msg_text[100];
		sprintf(msg_text, "%s: %s", msg, cudaGetErrorString(error));
		print_error(msg_text);
	}
}

void load_args(int argc, char **argv) {
	
	// Set size
	if (argc < 2) {
		print_error("you must specify size of test");	
	}
	size = atoi(argv[1]);
	print_info("setting size of init vector");

	// Set maximum number of steps
	if (argc < 3) {
		print_error("you must specify maximum number of simulation steps");
	}
	maximum_steps = atoi(argv[2]);
	print_info("setting maximum number of simulation steps");

	// Set step size
	if (argc < 4) {
		print_error("you must specify size of the time step");
	}
	step_size = atof(argv[3]);
	print_info("setting size of the time step");

	// Set target time
	if (argc < 5) {
		print_error("you must specify target time");
	}
	target_time = atof(argv[4]);
	print_info("setting target time");

	// Set dimension
	if (argc < 6) {
		print_error("you must specify the dimension");
	}
	vector_size = atof(argv[5]);
	print_info("setting dimension");

	// Set device
	device = 0;
	if (argc == 7)  {
		device = atoi(argv[6]);
	}
	char msg_text[100];
	sprintf(msg_text, "using device %i", device);
	print_info(msg_text);
	cudaSetDevice(device);
	check_cuda_error("cannot set CUDA device");

}

void init_data() {

	// init vector
	print_info("generating random init vector");
	cpu_init_vectors = init_increasing_vectors(size * vector_size);
	gpu_init_vectors = NULL;
	cudaMalloc((void**) &gpu_init_vectors, vector_size * size * sizeof(float));
	print_info("copying init vector to GPU");
	cudaMemcpy(gpu_init_vectors, cpu_init_vectors, vector_size * size* sizeof(float), cudaMemcpyHostToDevice);
	check_cuda_error("copying init vector to GPU failed");
	// memory for computed result
	print_info("preparing memory for computed resupt on GPU");
	gpu_result = NULL;
	cudaMalloc((void**) &gpu_result, vector_size * size * (ceil(target_time/step_size)) * sizeof(float));
	// number of executed steps
	print_info("preparing memory for number of executed steps on GPU");
	gpu_number_of_executed_steps = NULL;	
	cudaMalloc((void**) &gpu_number_of_executed_steps, size * sizeof(int));
	// return code
	print_info("preparing memory for return code on GPU");
	gpu_return_code = NULL;
	cudaMalloc((void**) &gpu_return_code, size * sizeof(int));
}

void execute() {
	int grid_size = ceil(sqrt(size / (BLOCK_SIZE * vector_size))) + 1;
	dim3 grid_dim(grid_size, grid_size);
	dim3 block_dim(BLOCK_SIZE, vector_size);
	char msg_text[100];
	sprintf(msg_text, "executing kernel (%i, %i) (%i, %i)", grid_size, grid_size, BLOCK_SIZE, vector_size);
	print_info(msg_text);
	rfk45_kernel<<<grid_dim,block_dim>>>(
		0,
		step_size,
		target_time,
		gpu_init_vectors,
		size,
		vector_size,
		0.00001,
		maximum_steps,
		gpu_result,
		gpu_number_of_executed_steps,
		gpu_return_code
	);
	check_cuda_error("executing kernel failed");
}

void load_result() {
	print_info("loading computed result");

	cpu_result = (float*) malloc(vector_size * size * (ceil(target_time/step_size)) * sizeof(float));
	cpu_number_of_executed_steps = (int*) malloc(size * sizeof(int));
	cpu_return_code = (int*) malloc(size * sizeof(int));

	cudaMemcpy(cpu_result, gpu_result, vector_size * size * (ceil(target_time/step_size)) * sizeof(float), cudaMemcpyDeviceToHost);
	check_cuda_error("copying result failed");
	cudaMemcpy(cpu_number_of_executed_steps, gpu_number_of_executed_steps, size * sizeof(int), cudaMemcpyDeviceToHost);
	check_cuda_error("copying number of executed steps failed");
	cudaMemcpy(cpu_return_code, gpu_return_code, size * sizeof(int), cudaMemcpyDeviceToHost);
	check_cuda_error("copying return code failed");
}

void print_result() {
	print_info("printing result");
	int max_target = ceil(target_time/step_size);
	for(int simulation_id = 0; simulation_id < size; simulation_id++) {
		int target = cpu_number_of_executed_steps[simulation_id];
		printf("\t\t---------------------------------------------------------\n");
		printf("\t\t[%i]\n", target);
		printf("\t\t---------------------------------------------------------\n");	
		printf("\t\t|");
		for (int i=0; i<vector_size; i++) {
			printf(" %4.2f\t|", cpu_init_vectors[simulation_id * vector_size + i]);
		}
		printf("\n");
		printf("\t\t---------------------------------------------------------\n");
		for (int step = 0; step < target; step++) {
			printf("\t\t|");
			for(int dimension = 0; dimension < vector_size; dimension++) {
				printf(" %4.2f\t|", cpu_result[max_target * vector_size * simulation_id + vector_size * step + dimension]);
			}
			printf("\n");
		}
	}
}

int main(int argc, char **argv) {
	load_args(argc, argv);
	init_data();
	execute();
	load_result();
	print_result();
	clean();
}
