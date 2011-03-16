package org.sybila.ode.cpu;

import org.sybila.ode.AbstractSimulationLauncher;
import org.sybila.ode.MultiAffineFunction;

abstract public class AbstractCpuSimulationLauncher extends AbstractSimulationLauncher
{

	public AbstractCpuSimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function);
	}

	public SimulationResult launch(
			float time,
			float timeStep,
			float targetTime,
			float[] vectors,
			int numberOfSimulations,
			float absDivergency) {

		org.sybila.ode.Simulation[] simulations = new Simulation[numberOfSimulations];
		for (int i = 0; i < numberOfSimulations; i++) {
			simulations[i] = simulate(i, time, timeStep, targetTime, vectors, numberOfSimulations, absDivergency);
		}
		return new org.sybila.ode.cpu.SimulationResult(simulations);
	}

	abstract protected org.sybila.ode.Simulation simulate(
			int simulationId,
			float time,
			float timeStep,
			float targetTime,
			float[] vectors,
			int numberOfSimulations,
			float absDivergency);

}
