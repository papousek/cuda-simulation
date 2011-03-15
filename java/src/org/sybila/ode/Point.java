package org.sybila.ode;

public interface Point
{

	int getDimension();

	float getValue(int index);

	float getTime();

	float[] toArray();

}
