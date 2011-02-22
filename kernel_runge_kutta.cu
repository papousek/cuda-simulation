float __device__ example_f(float t, float y) {
	return y;
}

void __global__ solve_ode(
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

	int id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	if (id >= size) return;
	float prev_y = init_y[id];
	unsigned int current_step = 0;		
	for(int i=0; i<(ceil(target_time/h)); i++) {
		current_step++;
		if (current_step > maximum_steps) {
			break;
		}
		float current_t = init_t + (i+1)*h;
		float k1 = h * example_f(current_t, prev_y);
		float k2 = h * example_f(current_t + h/2, prev_y + k1/2);
		float k3 = h * example_f(current_t + h/2, prev_y + k2/2);
		float k4 = h * example_f(current_t + h, prev_y + k3);	
		prev_y 	 = prev_y + 0.1666667 * (k1 + (2*k2) + (2*k3) + k4);
		result[size * i + id] = prev_y;
	}
	last_step[id] = current_step > maximum_steps ? maximum_steps : current_step;
}	
