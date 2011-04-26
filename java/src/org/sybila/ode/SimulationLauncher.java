package org.sybila.ode;

/**
 * The classes implementing this interface generate simulations.
 * 
 * @author Jan Papousek
 */
public interface SimulationLauncher
{

	/**
	 * It returns a set of simulations genereted order to the given parameters.
	 *
	 * @param time the time when the simulations (trajectories) are starting
	 * @param timeStep the maximum time step which the point differs from the previous one
	 * @param targetTime the time when the simulation should stop
	 * @param seeds seed the simulations are starting from (the value of dimension
	 * [d] and simulation [s] is located at the position seeds[s * s_size + d])
	 * @param numberOfSimulations number of simulations which will be executed
	 */
	public SimulationResult launch(
			float time,
			float timeStep,
			float targetTime,
			float[] seeds,
			int	 numberOfSimulations);

	public float getMaxAbsDivergency();

	public float getMaxRelDivergency();

	public float getMinAbsDivergency();

	public float getMinRelDivergency();

	public boolean hasMaxAbsDivergency();

	public boolean hasMaxRelDivergency();

	public boolean hasMinAbsDivergency();

	public boolean hasMinRelDivergency();



}
