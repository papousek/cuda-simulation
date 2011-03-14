package org.sybila.ode.cpu;

import org.sybila.ode.MultiAffineFunction;
import java.util.ArrayList;
import org.sybila.ode.Simulation;
import org.sybila.ode.SimulationLauncher;
import org.sybila.ode.SimulationResult;
import org.sybila.ode.SimulationStatus;

public class CpuSimulationLauncher implements SimulationLauncher {

	private static final float TIME_STEP = (float) 0.001;

	private MultiAffineFunction function;
	private int maxSimulationSize;
	private int numberOfSimulations;
	private int dimension;

	public CpuSimulationLauncher(int dimension, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
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

		Simulation[] simulations = new Simulation[numberOfSimulations];
		for (int i = 0; i < simulations.length; i++) {
			simulations[i] = simulate(i, time, timeStep, targetTime, vectors, absDivergency, maxNumberOfExecutedSteps);
		}
		return new org.sybila.ode.cpu.SimulationResult(simulations);
	}

	private Simulation simulate(
			int simulationId,
			float time,
			float timeStep,
			float targetTime,
			float[] vectors,
			float absDivergency,
			int maxNumberOfExecutedSteps) {

		ArrayList<Point> points = new ArrayList<Point>();
		float[] initData = new float[dimension];
		for (int i=0; i<dimension; i++) {
			initData[i] = vectors[simulationId * dimension + i];
		}
		Point previous = new Point(dimension, time, initData);
		float currentTime = time;
		int position = 0;
		for (int step=0; step<maxNumberOfExecutedSteps; step++) {
			currentTime += TIME_STEP;
			float[] data = new float[dimension];
			for(int dim=0; dim<dimension; dim++) {
				data[dim] = previous.getValue(dim) + TIME_STEP * function.compute(dim, previous);
			}
			previous = new Point(dimension, currentTime, data);
			if (currentTime >= (position+1) * timeStep) {
				position++;
				points.add(previous);
			}
			if (currentTime >= targetTime) break;
		}
		return new org.sybila.ode.cpu.Simulation(dimension, points.toArray(new Point[points.size()]), SimulationStatus.OK);
	}
}
