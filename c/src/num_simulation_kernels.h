typedef void (*t_ode_function) (
	const float, 	// time
	const float*, 	// vector
	const int, 	// size of vector
	const int,	// expected item of computed vector (0 => whole vector)
	float*,		// computed value(s)
);

enum SIMULATION_TYPE_ENUM {
	RUNGE_KUTTA_FEHLBERG,
	EULER
}

extern "C" solve_ode_on_gpu(
	/* INPUT */
	const enum SIMULATION_TYPE_ENUM simulation_type,
	const float*	init_vectors,
	const int	number_of_vectors,
	const int	size_of_vector,
	const float	init_time,
	const float 	target_time,
	const float	time_step,
	const int	max_number_of_steps,
	const float	abs_divergency,
	const float	abs_divergency,
	const t_ode_function ode_function,
	/* OUTPUT */
	float*		return_code,
	int*		number_of_successful_steps,
	float*		simulation
);
