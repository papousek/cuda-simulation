package org.sybila.ode.cuda;

import org.sybila.ode.Point;

public class NumericalSimulationResult
{

	private int[] numberOfExecutedSteps;

	private int numberOfSimulations;

	private int[] returnCodes;

	private float[] simulationTimes;

	private float[] simulationPoints;

	private int vectorSize;

	public NumericalSimulationResult(int numberOfSimulations, int vectorSize, int[] numberOfExecutedSteps, int[] returnCodes, float[] simulationTimes, float[] simulationPoints) {
		if (numberOfSimulations <= 0) {
			throw new IllegalArgumentException("The number of simulations has to be a positive number.");
		}
		if (vectorSize <= 0) {
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
		this.vectorSize				= vectorSize;
	}

	public int getNumberOfExecutedSteps(int simulationId) {
		if (simulationId < 0 || simulationId >= numberOfSimulations) {
			throw new IndexOutOfBoundsException("The simulation id [" + simulationId + "] is out of the range [0," + (numberOfSimulations-1) + "].");
		}
		return numberOfExecutedSteps[simulationId];
	}

	public int getNumberOfSimulations() {
		return numberOfSimulations;
	}

	public int getReturnCode(int simulationId) {
		if (simulationId < 0 || simulationId >= numberOfSimulations) {
			throw new IndexOutOfBoundsException("The simulation id [" + simulationId + "] is out of the range [0," + (numberOfSimulations-1) + "].");
		}
		return returnCodes[simulationId];
	}

//	public float getTime(int simulationId, int step) {
//		if (step < 0 || step >= getNumberOfExecutedSteps(simulationId)) {
//			throw new IndexOutOfBoundsException("The step [" + step + "] is out of the range [0," + (getNumberOfExecutedSteps(simulationId) - 1) + "].");
//		}
//		return simulationTimes[step * numberOfSimulations + simulationId];
//	}

	public Point getPoint(int simulationId, int step) {
		if (step < 0 || step >= getNumberOfExecutedSteps(simulationId)) {
			throw new IndexOutOfBoundsException("The step [" + step + "] is out of the range [0," + (getNumberOfExecutedSteps(simulationId) - 1) + "].");
		}
		int maxNumberOfSteps = simulationPoints.length / (vectorSize * numberOfSimulations);
		return new NumericalSimulationPoint(vectorSize, maxNumberOfSteps * vectorSize * simulationId + vectorSize * step, simulationPoints, simulationTimes[step * numberOfSimulations + simulationId]);
	}

	public int getVectorSize() {
		return vectorSize;
	}

}
