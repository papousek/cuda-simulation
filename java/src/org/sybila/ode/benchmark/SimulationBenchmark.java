package org.sybila.ode.benchmark;

import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.Simulation;
import org.sybila.ode.SimulationLauncher;
import org.sybila.ode.SimulationResult;
import org.sybila.ode.cpu.CpuEulerSimulationLauncher;

public class SimulationBenchmark {

	private int dimension;
	private static MultiAffineFunction function;
	private int maxSimulationSize;
	private int numberOfSimulations;
	private float[] vectors;

	public SimulationBenchmark(int dimension, int numberOfSimulations, int maxSimulationSize) {
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

	public static MultiAffineFunction createFunction(int dimension) {
		if (dimension <= 0) {
			throw new IllegalArgumentException("The parameter [dimension] has to be a positive number.");
		}
		float[] coefficients = new float[dimension];
		int[] coefficientIndexes = new int[dimension + 1];
		int[] factors = new int[dimension];
		int[] factorIndexes = new int[dimension + 1];

		for (int i = 0; i < coefficients.length; i++) {
			coefficients[i] = (float) 1;
		}

		for (int i = 0; i < coefficientIndexes.length; i++) {
			coefficientIndexes[i] = i;
		}

		for (int i = 0; i < factors.length; i++) {
			factors[i] = i;
		}

		for (int i = 0; i < factorIndexes.length; i++) {
			factorIndexes[i] = i;
		}

		function = new MultiAffineFunction(coefficients, coefficientIndexes, factors, factorIndexes);

		return function;
	}

	public SimulationBenchmarkResult bench(SimulationLauncher launcher) {
		long start = System.nanoTime();
		SimulationResult result = launcher.launch(0, 1, maxSimulationSize + 1, getVectors(), numberOfSimulations, (float) 0.01, (int) (maxSimulationSize / CpuEulerSimulationLauncher.TIME_STEP));
		long end = System.nanoTime();
		int sumLength = 0;
		for(int i=0; i<result.getNumberOfSimulations(); i++) {
			sumLength += result.getSimulation(i).getLength();
		}
		return new SimulationBenchmarkResult(end - start, sumLength / result.getNumberOfSimulations());
	}

	private float[] getVectors() {
		if (vectors == null) {
			vectors = new float[dimension * numberOfSimulations];
			for (int sim = 0; sim < numberOfSimulations; sim++) {
				for (int dim = 0; dim < dimension; dim++) {
					vectors[dimension * sim + dim] = 1;
				}
			}
		}
		return vectors;
	}
}
