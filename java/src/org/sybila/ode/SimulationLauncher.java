package org.sybila.ode;

public interface SimulationLauncher
{

	public SimulationResult launch(
			float time,
			float timeStep,
			float targetTime,
			float[] vectors,
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
