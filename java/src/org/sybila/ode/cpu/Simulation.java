package org.sybila.ode.cpu;

import org.sybila.ode.Point;
import org.sybila.ode.SimulationStatus;

public class Simulation implements org.sybila.ode.Simulation
{

	private int dimension;

	private Point[] points;

	private SimulationStatus status;

	public Simulation(int dimension, Point[] points, SimulationStatus status) {
		if (dimension <= 0) {
			throw new IllegalArgumentException("The size of vector has to be a positive number.");
		}
		if (points.length < 0) {
			throw new IllegalArgumentException("The lenght of the simulation has to be a non negative number.");
		}
		if (status == null) {
			throw new NullPointerException("The parameter [status] is NULL.");
		}
		this.dimension = dimension;
		this.points = points;
		this.status = status;
	}

	public int getDimension() {
		return dimension;
	}

	public Point getLastPoint() {
		return getPoint(getLength() - 1);
	}

	public int getLength() {
		return points.length;
	}

	public Point getPoint(int step) {
		if (step < 0 || step >= getLength()) {
			throw new IllegalArgumentException("The parameter [step] is out of the range [0, " + (getLength() - 1) + "].");
		}
		return points[step];
	}

	public SimulationStatus getStatus() {
		return status;
	}



}
