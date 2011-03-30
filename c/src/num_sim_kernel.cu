#define A2 (1.0/4.0)
#define B2 (1.0/4.0)

#define A3 (3.0/8.0)
#define B3 (3.0/32.0)
#define C3 (9.0/32.0)

#define	A4 (12.0/13.0)
#define B4 (1932.0/2197.0)
#define C4 (-7200.0/2197.0)
#define D4 (7296.0/2197.0)

#define A5 1.0
#define B5 (439.0/216.0)
#define C5 (-8.0)
#define D5 (3680.0/513.0)
#define E5 (-845.0/4104.0)

#define A6 (1.0/2.0)
#define B6 (-8.0/27.0)
#define C6 2.0
#define D6 (-3544.0/2565.0)
#define E6 (1859.0/4104.0)
#define F6 (-11.0/40.0)

#define R1 (1.0/360.0)
#define R3 (-128.0/4275.0)
#define R4 (-2197.0/75240.0)
#define R5 (1.0/50.0)
#define R6 (2.0/55.0)

#define N1 (25.0/216.0)
#define N3 (1408.0/2565.0)
#define N4 (2197.0/4104.0)
#define N5 (-1.0/5.0)

#define MINIMUM_TIME_STEP 0.0000001
#define MAXIMUM_TIME_STEP 100
#define MINIMUM_SCALAR_TO_OPTIMIZE_STEP 0.00001
#define MAXIMUM_SCALAR_TO_OPTIMIZE_STEP 4.0

#define STATUS_OK 0
#define STATUS_TIMEOUT 1
#define STATUS_PRECISION 2

__device__ void ode_function(
	const float 	time,
	const float* 	vector,
	const int	index,
	const int	vector_size,
	const float* 	coefficients,
	const int*	coefficient_indexes,
	const int*	factors,
	const int*	factor_indexes,
	float*		result
) {
	*result = 0;
	for(int c=coefficient_indexes[index]; c<coefficient_indexes[index+1]; c++) {
		float aux_result = coefficients[c];
		for(int f=factor_indexes[c]; f<factor_indexes[c+1]; f++) {
			aux_result *= vector[factors[f]];
		}
		*result += aux_result;
	}
}

extern "C"
void __global__ rkf45_kernel(
	const float 	time,
	const float 	time_step,
	const int	target_time,
	const float*	vectors,
	const int	number_of_vectors,
	const int	max_number_of_vectors,
	const int	vector_size,
	const float 	min_abs_divergency,
	const float 	max_abs_divergency,
	const float	min_rel_divergency,
	const float	max_rel_divergency,
	const int	max_number_of_steps,
	const int	simulation_max_size,
	const float* 	function_coefficients,
	const int*	function_coefficient_indexes,
	const int*	function_factors,
	const int*	function_factor_indexes,
	float* 		result_points,
	float*		result_times,
	int*		number_of_executed_steps,
	int*		return_code
) {
	// compute the number of simulations per block
	__shared__ int simulations_per_block;
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		simulations_per_block = (blockDim.x * blockDim.y) / vector_size;
	}
	__threadfence_block();
	__syncthreads();

	// compute the id within the block
	int id_in_block = threadIdx.y * blockDim.x + threadIdx.x;

	// if the thread can't compute any simulation, exit
	if (id_in_block >= vector_size * simulations_per_block) return;

	// compute index of simulation
	int simulation_id = (blockIdx.y * gridDim.x + blockIdx.x) * simulations_per_block + id_in_block / vector_size;

	// if the thread is out of the number of simulations, exit
	if (simulation_id >= number_of_vectors) return;

	// compute index of dimension
	int dimension_id  = id_in_block % vector_size;

	// reset number of executed steps and set the default time step
	*number_of_executed_steps = 0;	
	float h = time_step;

	// set current time and position
	float current_time 	= time;
	int position		= 0;

	// note the pointer on the vetor
	float* vector = &(result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * position]);

	// copy init vector to the result
	vector[dimension_id] = vectors[vector_size * simulation_id + dimension_id];

	// perform the simulation
	int steps = 0;
	while(steps < max_number_of_steps && current_time < target_time && position < simulation_max_size) {
		__threadfence_block();
		__syncthreads();
		steps++;
		float k1, k2, k3, k4, k5, k6, dim_value;
		dim_value = vector[dimension_id];
		// K1
		ode_function(current_time + h, vector, dimension_id, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &k1);
		k1 = k1 * h;
		// K2
		vector[dimension_id] = vector[dimension_id] + B2 * k1;
		__threadfence_block();
		__syncthreads();
		ode_function(current_time + A2 * h, vector, dimension_id, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &k2);
		k2 = k2 * h;
		// K3
		vector[dimension_id] = dim_value + B3 * k1 + C3 * k2;
		__threadfence_block();
		__syncthreads();
		ode_function(current_time + A3 * h, vector, dimension_id, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &k3);		
		k3 = k3 * h;
		// K4
		vector[dimension_id] = dim_value + B4 * k1 + C4 * k2 + D4 * k3;
		__threadfence_block();
		__syncthreads();
		ode_function(current_time + A4 * h, vector, dimension_id, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &k4);		
		k4 = k4 * h;
		// K5
		vector[dimension_id] = dim_value + B5 * k1 + C5 * k2 + D5 * k3 + E5 * k4;
		__threadfence_block();
		__syncthreads();
		ode_function(current_time + A5 * h, vector, dimension_id, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &k5);
		k5 = k5 * h;
		// K6
		vector[dimension_id] = dim_value + B6 * k1 + C6 * k2 + D6 * k3 + E6 * k4 + F6 * k5;
		__threadfence_block();	
		__syncthreads();
		ode_function(current_time + A6 * h, vector, dimension_id, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &k6);
		k6 = k6 * h;

		// reset vector
		__syncthreads();
		vector[dimension_id] = dim_value;
		__threadfence_block();
		__syncthreads();

		// error
		float error = abs(R1 * k1 + R3 * k3 + R4 * k4 + R5 * k5 + R6 * k6);
		__syncthreads();
		result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position+1) + dimension_id] = error;
		__threadfence_block();
		__syncthreads();
		error = 0;
		for (int i=0; i<vector_size; i++) {
			if (result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position+1) + i] > error) {
				error = result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position+1) + i];
			}
		}
		__syncthreads();

		// check absolute error
		if (max_abs_divergency > 0 && error >= max_abs_divergency) {
			// compute a new time step
			h /= 2;
			// precision has been lost
			if (h < MINIMUM_TIME_STEP) {
				return_code[simulation_id] = STATUS_PRECISION;
				number_of_executed_steps[simulation_id] = position;
				return;
			}
		}
		else {
			// result
			float dim_result = vector[dimension_id] + N1 * k1 + N3 * k3 + N4 * k4 + N5 * k5;
			// compute relative error
			float rel_error  = 0;
			result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position+1) + dimension_id] = error/dim_result;
			__threadfence_block();
			__syncthreads();
			// sync relative error
			if (max_rel_divergency > 0 && error > 0) {
				for(int i=0; i<vector_size; i++) {
					if (rel_error < result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position+1) + i]) {
						rel_error = result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position+1) + i];
					}
				}				
			}
			__syncthreads();
			// check relative error
			if (rel_error > max_rel_divergency) {
				// compute a new time step
				h /= 2;
				if (h < MINIMUM_TIME_STEP) {
					return_code[simulation_id] = STATUS_PRECISION;
					number_of_executed_steps[simulation_id] = position;
					return;
				}
			}
			// save result
			else {
				current_time += h;
				vector[dimension_id] = dim_result;
				if (current_time >= time_step * (position+1)) {
					result_times[simulation_id * simulation_max_size + position] = current_time;
					position++;
					vector = &(result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * position]);
					vector[dimension_id] = dim_result;
					if (error <= min_abs_divergency) {
						h *= 2;
						if (h > time_step) h = time_step;
					}
				}				
			}
		}
	}
	if (steps >= max_number_of_steps) return_code[simulation_id] = STATUS_TIMEOUT;
	else return_code[simulation_id] = STATUS_OK;
	number_of_executed_steps[simulation_id] = position;
}
