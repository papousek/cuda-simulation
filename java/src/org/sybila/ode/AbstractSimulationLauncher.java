package org.sybila.ode;

abstract public class AbstractSimulationLauncher implements SimulationLauncher
{

	public static final int MAX_NUMBER_OF_EXECUTED_STEPS = 100000;

	private int dimension;
	private MultiAffineFunction function;
	private float maxAbsDivergency;
	private float maxRelDivergency;
	private float minAbsDivergency;
	private float minRelDivergency;
	private int maxSimulationSize;
	private int maxNumberOfSimulations;

	public AbstractSimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		this(dimension, maxNumberOfSimulations, maxSimulationSize, function, (float) 0.0, (float) 0.0, (float) 0.0, (float) 0.001);
	}

	public AbstractSimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function, float minAbsDivergency, float maxAbsDivergency, float minRelDivergency, float maxRelDivergency) {
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
		this.maxAbsDivergency = maxAbsDivergency;
		this.maxRelDivergency = maxRelDivergency;
		this.minAbsDivergency = minAbsDivergency;
		this.minRelDivergency = minRelDivergency;
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

	public float getMaxAbsDivergency() {
		return maxAbsDivergency;
	}

	public float getMaxRelDivergency() {
		return maxRelDivergency;
	}

	public float getMinAbsDivergency() {
		return minAbsDivergency;
	}

	public float getMinRelDivergency() {
		return minRelDivergency;
	}

	public boolean hasMaxAbsDivergency() {
		return maxAbsDivergency > 0;
	}

	public boolean hasMaxRelDivergency() {
		return maxRelDivergency > 0;
	}

	public boolean hasMinAbsDivergency() {
		return minAbsDivergency > 0;
	}

	public boolean hasMinRelDivergency() {
		return minRelDivergency > 0;
	}
	
}
