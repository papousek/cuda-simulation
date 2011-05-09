package org.sybila.ode.cpu;

import org.sybila.ode.MultiAffineFunction;
import java.util.ArrayList;
import org.sybila.ode.Simulation;
import org.sybila.ode.SimulationStatus;

public class CpuEulerSimulationLauncher extends AbstractCpuSimulationLauncher
{

	public static final float TIME_STEP = (float) 0.01;

	public CpuEulerSimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function);
	}

	public CpuEulerSimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function, float minAbsDivergency, float maxAbsDivergency, float minRelDivergency, float maxRelDivergency) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function, minAbsDivergency, maxAbsDivergency, minRelDivergency, maxRelDivergency);
	}

	protected Simulation simulate(
			int simulationId,
			float time,
			float timeStep,
			float targetTime,
			float[] vectors,
			int numberOfSimulations) {

		ArrayList<Point> points = new ArrayList<Point>();
		float[] initData = new float[getDimension()];
		for (int i=0; i<getDimension(); i++) {
			initData[i] = vectors[simulationId * getDimension() + i];
		}
		Point previous = new Point(getDimension(), time, initData);
		float currentTime = time;
		int position = 0;
		float[] data = new float[getDimension()];
		for (int step=0; step<MAX_NUMBER_OF_EXECUTED_STEPS; step++) {
			currentTime += TIME_STEP;
			for(int dim=0; dim<getDimension(); dim++) {
				data[dim] = previous.getValue(dim) + TIME_STEP * getFunction().compute(dim, previous);
			}
			previous = new Point(getDimension(), currentTime, data);
			if (currentTime >= (position+1) * timeStep) {
				position++;
				points.add(previous);
			}
			if (currentTime >= targetTime) break;
			if (position >= getMaxSimulationSize()) break;
		}
		return new org.sybila.ode.cpu.Simulation(getDimension(), points.toArray(new Point[points.size()]), SimulationStatus.OK);
	}
}
