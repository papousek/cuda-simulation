package org.sybila.ode.cuda;

import java.util.Arrays;
import org.sybila.ode.Simulation;

public class SimulationResult implements org.sybila.ode.SimulationResult
{

	private int[] numberOfExecutedSteps;

	private int numberOfSimulations;

	private int[] returnCodes;

	private float[] simulationTimes;

	private float[] simulationPoints;

	private int dimension;

	public SimulationResult(int numberOfSimulations, int dimension, int[] numberOfExecutedSteps, int[] returnCodes, float[] simulationTimes, float[] simulationPoints) {
		if (numberOfSimulations <= 0) {
			throw new IllegalArgumentException("The number of simulations has to be a positive number.");
		}
		if (dimension <= 0) {
			throw new IllegalArgumentException("The size of vector has to be a positive number.");
		}
		if (numberOfExecutedSteps == null) {
			throw new NullPointerException("The parameter [numberOfExecutedSteps] is NULL.");
		}
		if (numberOfExecutedSteps.length != numberOfSimulations) {
			throw new IllegalArgumentException("The length of the array containg number of executed steps doesn't correspond to the number of simulations.");
		}
		if (returnCodes == null) {
			throw new NullPointerException("The parameter [returnCodes] is NULL.");
		}
		if (numberOfExecutedSteps.length != numberOfSimulations) {
			throw new IllegalArgumentException("The length of the array containg return codes doesn't correspond to the number of simulations.");
		}
		if (simulationTimes == null) {
			throw new NullPointerException("The parameter [simulationTimes] is NULL.");
		}
		if (numberOfExecutedSteps.length % numberOfSimulations != 0) {
			throw new IllegalArgumentException("The length of the array containg simulation times doesn't correspond to the number of simulations.");
		}
		if (simulationPoints == null) {
			throw new NullPointerException("The parameter [simulationPoints] is NULL.");
		}
		this.numberOfExecutedSteps	= numberOfExecutedSteps;
		this.numberOfSimulations	= numberOfSimulations;
		this.returnCodes			= returnCodes;
		this.simulationPoints		= simulationPoints;
		this.simulationTimes		= simulationTimes;
		this.dimension				= dimension;
	}

	public int getNumberOfSimulations() {
		return numberOfSimulations;
	}

	public Simulation getSimulation(int simulationId) {
		if (simulationId < 0 || simulationId >= getNumberOfSimulations()) {
			throw new IllegalArgumentException("The simulation ID is out of the range [0," + getNumberOfSimulations() + "].");
		}
		int maxNumberOfSteps = simulationPoints.length / (dimension * numberOfSimulations);
		return new org.sybila.ode.cuda.Simulation(dimension, numberOfExecutedSteps[simulationId], returnCodes[simulationId], simulationId, maxNumberOfSteps, numberOfSimulations, simulationPoints, simulationTimes);
	}

}
