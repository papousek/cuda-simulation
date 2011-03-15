package org.sybila.ode.cpu;

import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.SimulationLauncher;

abstract public class AbstractCpuSimulationLauncher implements SimulationLauncher
{

	protected static final float TIME_STEP = (float) 0.001;

	private MultiAffineFunction function;
	private int maxSimulationSize;
	private int numberOfSimulations;
	private int dimension;

	public AbstractCpuSimulationLauncher(int dimension, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		if (dimension <= 0) {
			throw new IllegalArgumentException("The parameter [vectorSize] has to be a positive number.");
		}
		if (numberOfSimulations <= 0) {
			throw new IllegalArgumentException("The parameter [numberOfSimulations] has to be a positive number.");
		}
		if (maxSimulationSize <= 0) {
			throw new IllegalArgumentException("The parameter [maxSimulationSize] has to be a positive number.");
		}
		if (function == null) {
			throw new NullPointerException("The parameter [function] is NULL.");
		}
		this.dimension = dimension;
		this.maxSimulationSize = maxSimulationSize;
		this.numberOfSimulations = numberOfSimulations;
		this.function = function;
	}

	public SimulationResult launch(
			float time,
			float timeStep,
			float targetTime,
			float[] vectors,
			float absDivergency,
			int maxNumberOfExecutedSteps) {

		org.sybila.ode.Simulation[] simulations = new Simulation[numberOfSimulations];
		for (int i = 0; i < simulations.length; i++) {
			simulations[i] = simulate(i, time, timeStep, targetTime, vectors, absDivergency, maxNumberOfExecutedSteps);
		}
		return new org.sybila.ode.cpu.SimulationResult(simulations);
	}

	abstract protected org.sybila.ode.Simulation simulate(
			int simulationId,
			float time,
			float timeStep,
			float targetTime,
			float[] vectors,
			float absDivergency,
			int maxNumberOfExecutedSteps);

	protected MultiAffineFunction getFunction() {
		return function;
	}

	protected int getDimension() {
		return dimension;
	}

	protected int getMaxSimulationSize() {
		return maxSimulationSize;
	}

	protected int getNumberOfSimulations() {
		return numberOfSimulations;
	}



}
