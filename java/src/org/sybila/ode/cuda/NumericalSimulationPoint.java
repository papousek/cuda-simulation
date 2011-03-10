package org.sybila.ode.cuda;

import org.sybila.ode.Point;

public class NumericalSimulationPoint implements Point
{

	private int dimension;

	private float[] data;

	private int startIndex;

	private float time;

	public NumericalSimulationPoint(int dimension, int startIndex, float[] data, float time) {
		if (dimension <= 0) {
			throw new IllegalArgumentException("The dimension of the point has to be a positive number.");
		}
		if (data == null) {
			throw new NullPointerException("The parameter [data] is NULL.");
		}
		if (startIndex < 0 || startIndex > data.length - dimension) {
			throw new IndexOutOfBoundsException("The index [" + startIndex + "] is out of the range [0," + (data.length - dimension) + "].");
		}
		this.dimension	= dimension;
		this.data		= data;
		this.startIndex	= startIndex;
	}

	public int getDimension() {
		return dimension;
	}

	public float getTime() {
		return time;
	}

	public float getValue(int index) {
		if (index < 0 || index >= dimension) {
			throw new IndexOutOfBoundsException("The index [" + startIndex + "] is out of the range [0," + dimension + "].");
		}
		return data[startIndex + index];
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof Point)) {
			return false;
		}
		Point p = (Point) o;
		if (p.getDimension() != this.getDimension()) {
			return false;
		}
		for(int i=0; i<getDimension(); i++) {
			if (p.getValue(i) != this.getValue(i)) {
				return false;
			}
		}
		return true;
	}

	@Override
	public int hashCode() {
		int hash = 7;
		hash = 43 * hash + this.dimension;
		return hash;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("[");
		boolean notFirst = false;
		for(int i=startIndex; i<startIndex+dimension; i++) {
			if (notFirst) {
				builder.append(", " + data[i]);
			}
			else {
				builder.append(data[i]);
				notFirst = true;
			}
		}
		builder.append("]");
		return builder.toString();
	}

}
