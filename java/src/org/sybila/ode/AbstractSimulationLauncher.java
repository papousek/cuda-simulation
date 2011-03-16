package org.sybila.ode;

abstract public class AbstractSimulationLauncher implements SimulationLauncher
{

	public static final int MAX_NUMBER_OF_EXECUTED_STEPS = 100000;

	private MultiAffineFunction function;
	private int maxSimulationSize;
	private int maxNumberOfSimulations;
	private int dimension;

	public AbstractSimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		if (dimension <= 0) {
			throw new IllegalArgumentException("The parameter [dimension] has to be a positive number.");
		}
		if (maxNumberOfSimulations <= 0) {
			throw new IllegalArgumentException("The parameter [maxNumberOfSimulations] has to be a positive number.");
		}
		if (maxSimulationSize <= 0) {
			throw new IllegalArgumentException("The parameter [maxSimulationSize] has to be a positive number.");
		}
		if (function == null) {
			throw new NullPointerException("The parameter [function] is NULL.");
		}
		this.dimension = dimension;
		this.maxSimulationSize = maxSimulationSize;
		this.maxNumberOfSimulations = maxNumberOfSimulations;
		this.function = function;
	}

	protected MultiAffineFunction getFunction() {
		return function;
	}

	public int getDimension() {
		return dimension;
	}

	public int getMaxNumberOfSimulations() {
		return maxNumberOfSimulations;
	}

	public int getMaxSimulationSize() {
		return maxSimulationSize;
	}

}
