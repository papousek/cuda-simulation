package org.sybila.ode;

public interface Simulation
{

	int getDimension();

	Point getLastPoint();

	int getLength();

	Point getPoint(int step);

	SimulationStatus getStatus();
}
