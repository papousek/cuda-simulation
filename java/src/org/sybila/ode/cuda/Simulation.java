package org.sybila.ode.cuda;

import org.sybila.ode.Point;
import org.sybila.ode.SimulationStatus;

public class Simulation implements org.sybila.ode.Simulation
{

	private float[] data;

	private int dimension;

	private int id;

	private int length;

	private int maxLength;

	private int numberOfSimulations;

	private SimulationStatus status;

	private float[] times;

	public Simulation(int dimension, int length, int status, int id, int maxLength, int numberOfSimulations, float[] data, float[] times) {
		if (dimension <= 0) {
			throw new IllegalArgumentException("The size of vector has to be a positive number.");
		}
		if (length < 0) {
			throw new IllegalArgumentException("The lenght of the simulation has to be a non negative number.");
		}
		if (id < 0) {
			throw new IllegalArgumentException("The simulation ID has to be a non negative number.");
		}
		if (maxLength <= 0) {
			throw new IllegalArgumentException("The maximum lenght of the simulation has to be a positive number.");
		}
		if (numberOfSimulations <= 0) {
			throw new IllegalArgumentException("The number of simulations has to be a positive number.");
		}
		if (data == null) {
			throw new NullPointerException("The parameter [data] is NULL.");
		}
		if (times == null) {
			throw new NullPointerException("The parameter [times] is NULL.");
		}
		this.data = data;
		this.dimension = dimension;
		this.id = id;
		this.length = length;
		this.maxLength = maxLength;
		this.status = SimulationStatus.fromInt(status);
		this.times = times;
		this.numberOfSimulations = numberOfSimulations;
	}

	public int getDimension() {
		return dimension;
	}

	public Point getLastPoint() {
		if (length != 0) {
			return getPoint(getLength() - 1);
		}
		else {
			return null;
		}
	}

	public int getLength() {
		return length;
	}

	public Point getPoint(int step) {
		if (step < 0 || step >= getLength()) {
			throw new IllegalArgumentException("The parameter [step] is out of the range [0, " + (getLength() - 1) + "].");
		}
		return new org.sybila.ode.cuda.Point(dimension, maxLength * dimension * id + dimension * step, data, times[id * maxLength + step]);
	}

	public SimulationStatus getStatus() {
		return status;
	}

}
