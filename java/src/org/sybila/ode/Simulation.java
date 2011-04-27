package org.sybila.ode;

/**
 * The sequence of points and times which is produced by numerical method
 *
 * @author Jan Papousek
 */
public interface Simulation
{

	/**
	 * It returns number of values which contains one point of the simulation
	 *
	 */
	int getDimension();

	/**
	 * It returns the last point of the simulation
	 */
	Point getLastPoint();

	/**
	 * It returns number of points which the simulation contains
	 */
	int getLength();

	/**
	 * It returns a point by the given index (starting from 0)
	 *
	 * @param step index of the point
	 * @throws IllegalArgumentException if the index is out of the range
	 */
	Point getPoint(int step);

	/**
	 * It returns a simulation status
	 */
	SimulationStatus getStatus();
}
