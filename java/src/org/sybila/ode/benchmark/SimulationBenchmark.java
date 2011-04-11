package org.sybila.ode.benchmark;

import java.util.ArrayList;
import java.util.Collection;
import jcuda.CudaException;
import org.sybila.ode.Equation;
import org.sybila.ode.EquationSystem;
import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.Simulation;
import org.sybila.ode.SimulationLauncher;
import org.sybila.ode.SimulationResult;
import org.sybila.ode.Variable;

public class SimulationBenchmark {

	public static final float TIME_STEP = (float) 0.1;

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

	public static EquationSystem createSimpleFunction(int dimension) {
		Collection<Equation> eqs = new ArrayList<Equation>();
		for(int i=0; i<dimension; i++) {
			Variable var = Variable.generate();
			eqs.add(Equation.parse("d" + var + " = 0.005 * " + var));
		}
		return new EquationSystem(eqs);
	}

	public static EquationSystem createLinearFunction(int dimension) {
		if (dimension == 1) {
			return createSimpleFunction(dimension);
		}
		Collection<Equation> eqs = new ArrayList<Equation>();
		Variable[] vars = new Variable[dimension];
		for(int i=0; i<dimension; i++) {
			vars[i] = Variable.generate();
		}
		for(int i=0; i<dimension; i++) {
			eqs.add(Equation.parse("d" + vars[i] + " = " + (0.01 * (i+1)) + " * " + vars[i] + " + " + (0.01 * dimension - i * 0.1) + " * " + vars[(i+1) % dimension]));
		}
		return new EquationSystem(eqs);
	}

	public static EquationSystem createLotkaVolteraFunction(int dimension) {
		if (dimension != 2) {
			throw new IllegalArgumentException();
		}
		Collection<Equation> eqs = new ArrayList<Equation>();
		Variable v1 = Variable.generate();
		Variable v2 = Variable.generate();
		eqs.add(Equation.parse("d" + v1 + " = 10.1*" + v1 + " + (-1.0)*" + v1 + "*" + v2));
		eqs.add(Equation.parse("d" + v2 + " = 10.1*" + v1 + " * " + v2 + " + (-5.4)*" + v2));
		return new EquationSystem(eqs);
	}

	public SimulationBenchmarkResult bench(SimulationLauncher launcher) {
		try {
			long start = System.nanoTime();
			SimulationResult result = launcher.launch(0, TIME_STEP, TIME_STEP * maxSimulationSize + TIME_STEP, getVectors(), numberOfSimulations);
			long end = System.nanoTime();
			int sumLength = 0;
			for(int i=0; i<result.getNumberOfSimulations(); i++) {
				sumLength += result.getSimulation(i).getLength();
			}
			return new SimulationBenchmarkResult(end - start, sumLength / result.getNumberOfSimulations());
		}
		catch(CudaException e) {
//			System.err.println(e.getMessage());
			return new SimulationBenchmarkResult(0, 0);
		}
		
	}

	private float[] getVectors() {
		if (vectors == null) {
			vectors = new float[dimension * numberOfSimulations];
			for (int sim = 0; sim < numberOfSimulations; sim++) {
				for (int dim = 0; dim < dimension; dim++) {
					vectors[dimension * sim + dim] = dimension * dim + sim;
				}
			}
		}
		return vectors;
	}
}
