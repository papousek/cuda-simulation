#define CODE_TIMEOUT 1
#define CODE_SUCCESS 2


#define A2 (1/4)
#define B2 (1/4)

#define A3 (3/8)
#define B3 (3/32)
#define C3 (9/32)

#define	A4 (12/13)
#define B4 (1932/2197)
#define C4 (-7200/2197)
#define D4 (7296/2197)

#define A5 1
#define B5 (439/216)
#define C5 (-8)
#define D5 (3680/513)
#define E5 (-845/4104)

#define A6 (1/2)
#define B6 (-8/27)
#define C6 2
#define D6 (-3544/2565)
#define E6 (1859/4104)
#define F6 (-11/40)

#define R1 (1/360)
#define R3 (-128/4275)
#define R4 (-2197/75240)
#define R6 (2/55)

#define N1 (25/216)
#define N3 (1408/2565)
#define N4 (2197/4104)
#define N5 (-1/5)

float __device__ example_f(float t, float y) {
	return y;
}

void __global__ rfk45_kernel(
	/* INPUT */
	const enum SIMULATION_TYPE_ENUM simulation_type,
	const float* init_vectors,
	const int number_of_vectors,
	const int size_of_vector,
	const float init_time,
	const float target_time,
	const float time_step,
	const int max_number_of_steps,
	const float abs_divergency,
	const float abs_divergency,
	/* OUTPUT */
	float* return_code,
	int* number_of_successful_steps,
	float* simulation
) {

	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (id >= number_of_vectors) return;

	float* previous_vector = init_vectors[id * number_of_vectors];
	unsigned int current_step = 0;

}
