package org.sybila.ode.benchmark;

import java.util.ArrayList;
import java.util.Collection;
import org.sybila.ode.Equation;
import org.sybila.ode.EquationSystem;
import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.Simulation;
import org.sybila.ode.SimulationLauncher;
import org.sybila.ode.SimulationResult;
import org.sybila.ode.Variable;

public class SimulationBenchmark {

	public static final float TIME_STEP = (float) 1;

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

	public static MultiAffineFunction createSimpleFunction(int dimension) {
		Collection<Equation> eqs = new ArrayList<Equation>();
		for(int i=0; i<dimension; i++) {
			Variable var = Variable.generate();
			eqs.add(Equation.parse("d" + var + " = 0.005 * " + var));
		}
		return new EquationSystem(eqs).getFunction();
	}

	public SimulationBenchmarkResult bench(SimulationLauncher launcher) {
		long start = System.nanoTime();
		SimulationResult result = launcher.launch(0, TIME_STEP, TIME_STEP * maxSimulationSize + TIME_STEP, getVectors(), numberOfSimulations);
		long end = System.nanoTime();
		int sumLength = 0;
//		for(int i=0; i<result.getNumberOfSimulations(); i++) {
////			System.out.println(result.getSimulation(i).getLength());
//			sumLength += result.getSimulation(i).getLength();
//		}
//		Simulation sim = result.getSimulation(0);
//		System.out.println("[" + sim.getStatus() + "]");
//
//		for (int i=0; i<(sim.getLength() < 10 ? sim.getLength() : 10); i++) {
//			System.out.println(sim.getPoint(i));
//		}
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
