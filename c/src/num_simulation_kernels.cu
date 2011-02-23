#include "rfk45.cu"

void solve_ode_on_gpu(
	/* INPUT */
	const float*	init_vectors,
	const int	number_of_vectors,
	const int	size_of_vector,
	const float	init_time,
	const float 	target_time,
	const float	time_step,
	const int	max_number_of_steps,
	const float	abs_divergency,
	const float	rel_divergency,
	const t_ode_function ode_function,
	/* OUTPUT */
	float*		return_code,
	int*		number_of_successful_steps,
	float*		simulation
) {
}
