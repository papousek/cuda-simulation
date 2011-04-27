package org.sybila.ode;

/**
 * One point of the trajectory produced by the numerical method
 *
 * @author Jan Papousek
 */
public interface Point
{

	/**
	 * It returns number of values which the point contains
	 */
	int getDimension();

	/**
	 * It returns a value with the specified index (starting from 0)
	 * @throws IllegalArgumentException if the index is out of the range
	 */
	float getValue(int index);

	/**
	 * It returns the time of the point
	 */
	float getTime();

	/**
	 * It returns an array representation of the point
	 */
	float[] toArray();

}
