package org.sybila.ode.benchmark;

import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.SimulationLauncher;

public class SimulationBenchmark
{

	private int dimension;

	private MultiAffineFunction function;

	private int maxSimulationSize;

	private int numberOfSimulations;

	private float[] vectors;

	public SimulationBenchmark(int dimension, int numberOfSimulations, int maxSimulationSize) {
		if (dimension <= 0) {
			throw new IllegalArgumentException("The parameter [vectorSize] has to be a positive number.");
		}
		if (numberOfSimulations <= 0) {
			throw new IllegalArgumentException("The parameter [numberOfSimulations] has to be a positive number.");
		}
		if (maxSimulationSize <= 0) {
			throw new IllegalArgumentException("The parameter [maxSimulationSize] has to be a positive number.");
		}
		this.dimension = dimension;
		this.maxSimulationSize = maxSimulationSize;
		this.numberOfSimulations = numberOfSimulations;
	}

	public MultiAffineFunction getFunction() {
		if (function == null) {
			float[] coefficients = new float[dimension];
			int[] coefficientIndexes = new int[dimension+1];
			int[] factors = new int[dimension];
			int[] factorIndexes = new int[dimension+1];

			for(int i=0; i<coefficients.length; i++) {
				coefficients[i] = (float) 0.005;
			}

			for(int i=0; i<coefficientIndexes.length; i++) {
				coefficientIndexes[i] = i;
			}

			for(int i=0; i<factors.length; i++) {
				factors[i] = i;
			}

			for(int i=0; i<factorIndexes.length; i++) {
				factorIndexes[i] = i;
			}

			function = new MultiAffineFunction(coefficients, coefficientIndexes, factors, factorIndexes);
		}
		return function;
	}

	public long time(SimulationLauncher launcher) {
		long start = System.nanoTime();
		launcher.launch(0, 1, maxSimulationSize + 1, getVectors(), (float) 0.01, maxSimulationSize);
		long end = System.nanoTime();
		return (end - start);
	}

	private float[] getVectors() {
		if (vectors == null) {
			vectors = new float[dimension * numberOfSimulations];
			for(int sim=0; sim<numberOfSimulations; sim++) {
				for(int dim=0; dim<dimension; dim++) {
					vectors[dimension * sim + dim] = dim+1;
				}
			}
		}
		return vectors;
	}

}
