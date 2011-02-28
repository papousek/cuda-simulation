#include "num_simulation_kernels.cu"

extern void __global__ rfk45_kernel(
	const float time,
	const float 		time_step,
	const float*		vector,
	const int		vector_size,
	const int 		variable_index,
	const float 		abs_divergency,
	const int		max_number_of_steps,
	float* 			result,
	int*			number_of_executed_steps,
	int*			return_code
);
