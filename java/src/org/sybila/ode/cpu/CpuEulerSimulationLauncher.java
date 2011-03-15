package org.sybila.ode.cpu;

import org.sybila.ode.MultiAffineFunction;
import java.util.ArrayList;
import org.sybila.ode.Simulation;
import org.sybila.ode.SimulationStatus;

public class CpuEulerSimulationLauncher extends AbstractCpuSimulationLauncher {


	public CpuEulerSimulationLauncher(int dimension, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		super(dimension, numberOfSimulations, maxSimulationSize, function);
	}

	protected Simulation simulate(
			int simulationId,
			float time,
			float timeStep,
			float targetTime,
			float[] vectors,
			float absDivergency,
			int maxNumberOfExecutedSteps) {

		ArrayList<Point> points = new ArrayList<Point>();
		float[] initData = new float[getDimension()];
		for (int i=0; i<getDimension(); i++) {
			initData[i] = vectors[simulationId * getDimension() + i];
		}
		Point previous = new Point(getDimension(), time, initData);
		float currentTime = time;
		int position = 0;
		for (int step=0; step<maxNumberOfExecutedSteps; step++) {
			currentTime += TIME_STEP;
			float[] data = new float[getDimension()];
			for(int dim=0; dim<getDimension(); dim++) {
				data[dim] = previous.getValue(dim) + TIME_STEP * getFunction().compute(dim, previous);
			}
			previous = new Point(getDimension(), currentTime, data);
			if (currentTime >= (position+1) * timeStep) {
				position++;
				points.add(previous);
			}
			if (currentTime >= targetTime) break;
		}
		return new org.sybila.ode.cpu.Simulation(getDimension(), points.toArray(new Point[points.size()]), SimulationStatus.OK);
	}
}
