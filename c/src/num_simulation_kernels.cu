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

#define MINIMUM_TIME_STEP 0.000001
#define MAXIMUM_TIME_STEP 100
#define MINIMUM_SCALAR_TO_OPTIMIZE_STEP 0.01
#define MAXIMUM_SCALAR_TO_OPTIMIZE_STEP 4.0

#ifndef KERNEL_RETURN_CODES
#define KERNEL_RETURN_CODES
#define CODE_TIMEOUT 1
#define CODE_SUCCESS 2
#define CODE_PRECISION_FAILED 3
#endif

inline void __device__ example_f(const float time, const float* vector, const int index, const int vector_size, float* result) {
	*result = vector[index];
}

void __global__ rfk45_kernel(
	const float time,
	const float 		time_step,
	const int		target_time,
	const float*		vectors,
	const int		number_of_vectors,
	const int		vector_size,
	const float 		abs_divergency,
	void (*ode_function) (const float, const float*, const int, const int, float*),
	const int		max_number_of_steps,
	float* 			result,
	int*			number_of_executed_steps,
	int*			return_code
) {

	// compute the number of simulations per block
	__shared__ int simulations_per_block;
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		simulations_per_block = blockDim.x * blockDim.y / vector_size;
	}
	__syncthreads();
	
	// compute the id within the block
	int id_in_block = threadIdx.y * blockDim.x + threadIdx.x;

	// if the thread can't compute any simulation, exit
	if (id_in_block >= vector_size * simulations_per_block) return;

	// compute index of simulation
	int simulation_id = (blockIdx.y * gridDim.x + blockIdx.y) * simulations_per_block + id_in_block / vector_size;

	// if the thread is out of the number of simulations, exit
	if (simulation_id > number_of_vectors) return;

	// compute index of dimension
	int dimension_id  = id_in_block % vector_size;

	// reset number of executed steps and set the default time step
	*number_of_executed_steps = 0;	
	float h = time_step;
	
	// compute max position in time
	__shared__ int max_position;
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		max_position = ceil((target_time - time)/time_step);
	}
	__syncthreads();

	// set current time and position
	float current_time 	= time;
	int position		= 0;

	// note the pointer on the vetor
	const float* vector 	= &(vectors[vector_size * simulation_id]);

	// perform the simulation
	while(position < max_number_of_steps && current_time < target_time) {
		float k1, k2, k3, k4, k5, k6;
		// K1
		example_f(current_time + h, vector, dimension_id, vector_size, &k1);
		k1 *= h;
		// K2
		result[max_position * vector_size * simulation_id  + vector_size * position + dimension_id] = vector[dimension_id] + B2 * k1;
		__syncthreads();
		example_f(current_time + A2 * h, &(result[max_position * vector_size * simulation_id  + vector_size * position]), dimension_id, vector_size, &k2);
		k2 *= h;
		// K3
		result[max_position * vector_size * simulation_id  + vector_size * position + dimension_id] = vector[dimension_id] + B3 * k1 + C3 * k2;
		__syncthreads();
		example_f(current_time + A3 * h, &(result[max_position * vector_size * simulation_id  + vector_size * position]), dimension_id, vector_size, &k3);		
		k3 *= h;
		// K4
		result[max_position * vector_size * simulation_id  + vector_size * position + dimension_id] = vector[dimension_id] + B4 * k1 + C4 * k2 + D4 * k3;
		__syncthreads();
		example_f(current_time + A4 * h, &(result[max_position * vector_size * simulation_id  + vector_size * position]), dimension_id, vector_size, &k4);		
		k4 *= h;
		// K5
		result[max_position * vector_size * simulation_id  + vector_size * position + dimension_id] = vector[dimension_id] + B5 * k1 + C5 * k2 + D5 * k3 + E5 * k4;
		__syncthreads();
		example_f(current_time + A5 * h, &(result[max_position * vector_size * simulation_id  + vector_size * position]), dimension_id, vector_size, &k5);
		k5 *= h;

		// result
		result[max_position * vector_size * simulation_id  + vector_size * position + dimension_id] = vector[dimension_id] + N1 * k1 + N3 * k3 + N4 * k4 + N5 * k5;
		
		// update time and position
		vector = &(result[max_position * vector_size * simulation_id  + vector_size * position]);
		current_time += h;
		position++;
	}
	number_of_executed_steps[simulation_id] = position;
}
