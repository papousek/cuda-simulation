package org.sybila.ode.cpu;

import java.util.ArrayList;
import java.util.List;
import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.Simulation;
import org.sybila.ode.SimulationStatus;

public class CpuRkf45SimulationLauncher extends AbstractCpuSimulationLauncher {

	private static final float A2 = (float) (1.0 / 4.0);
	private static final float B2 = (float) (1.0 / 4.0);
	private static final float A3 = (float) (3.0 / 8.0);
	private static final float B3 = (float) (3.0 / 32.0);
	private static final float C3 = (float) (9.0 / 32.0);
	private static final float A4 = (float) (12.0 / 13.0);
	private static final float B4 = (float) (1932.0 / 2197.0);
	private static final float C4 = (float) (-7200.0 / 2197.0);
	private static final float D4 = (float) (7296.0 / 2197.0);
	private static final float A5 = (float) 1.0;
	private static final float B5 = (float) (439.0 / 216.0);
	private static final float C5 = (float) (-8.0);
	private static final float D5 = (float) (3680.0 / 513.0);
	private static final float E5 = (float) (-845.0 / 4104.0);
	private static final float A6 = (float) (1.0 / 2.0);
	private static final float B6 = (float) (-8.0 / 27.0);
	private static final float C6 = (float) 2.0;
	private static final float D6 = (float) (-3544.0 / 2565.0);
	private static final float E6 = (float) (1859.0 / 4104.0);
	private static final float F6 = (float) (-11.0 / 40.0);
	private static final float R1 = (float) (1.0 / 360.0);
	private static final float R3 = (float) (-128.0 / 4275.0);
	private static final float R4 = (float) (-2197.0 / 75240.0);
	private static final float R5 = (float) (1.0 / 50.0);
	private static final float R6 = (float) (2.0 / 55.0);
	private static final float N1 = (float) (25.0 / 216.0);
	private static final float N3 = (float) (1408.0 / 2565.0);
	private static final float N4 = (float) (2197.0 / 4104.0);
	private static final float N5 = (float) (-1.0 / 5.0);
	private static final float MINIMUM_TIME_STEP = (float) 0.00000000001;
	private static final float MAXIMUM_TIME_STEP = (float) 100;
	private static final float MINIMUM_SCALAR_TO_OPTIMIZE_STEP = (float) 0.01;
	private static final float MAXIMUM_SCALAR_TO_OPTIMIZE_STEP = (float) 4.0;

	public CpuRkf45SimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function);
	}

	public CpuRkf45SimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function, float minAbsDivergency, float maxAbsDivergency, float minRelDivergency, float maxRelDivergency) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function, minAbsDivergency, maxAbsDivergency, minRelDivergency, maxRelDivergency);
	}

	@Override
	protected Simulation simulate(int simulationId, float time, float timeStep, float targetTime, float[] vectors, int numberOfSimulations) {

		List<Point> points = new ArrayList<Point>();
		float[] initData = new float[getDimension()];
		for (int i = 0; i < getDimension(); i++) {
			initData[i] = vectors[simulationId * getDimension() + i];
		}
		Point previous = new Point(getDimension(), time, initData);

		float[] k1 = new float[getDimension()];
		float[] k2 = new float[getDimension()];
		float[] k3 = new float[getDimension()];
		float[] k4 = new float[getDimension()];
		float[] k5 = new float[getDimension()];
		float[] k6 = new float[getDimension()];

		float currentTime = time;
		int position = 0;
		float myStep = timeStep;
		float[] localError = new float[getDimension()];
		float[] data = previous.toArray();
		
		for (int step = 0; step < MAX_NUMBER_OF_EXECUTED_STEPS; step++) {
			float error = 0;
			// K1
			for (int dim = 0; dim < getDimension(); dim++) {
				k1[dim] = myStep * getFunction().compute(dim, data);
			}
			// K2
			for (int dim = 0; dim < getDimension(); dim++) {
				data[dim] = data[dim] + B2 * k1[dim];
			}
			for (int dim = 0; dim < getDimension(); dim++) {
				k2[dim] = myStep * getFunction().compute(dim, data);
			}
			// K3
			for (int dim = 0; dim < getDimension(); dim++) {
				data[dim] = data[dim] + B3 * k1[dim] + C3 * k2[dim];
			}
			for (int dim = 0; dim < getDimension(); dim++) {
				k3[dim] = myStep * getFunction().compute(dim, data);
			}
			// K4
			for (int dim = 0; dim < getDimension(); dim++) {
				data[dim] = data[dim] + B4 * k1[dim] + C4 * k2[dim] + D4 * k3[dim];
			}
			for (int dim = 0; dim < getDimension(); dim++) {
				k4[dim] = myStep * getFunction().compute(dim, data);
			}
			// K5
			for (int dim = 0; dim < getDimension(); dim++) {
				data[dim] = data[dim] + B5 * k1[dim] + C5 * k2[dim] + D5 * k3[dim] + E5 * k4[dim];
			}
			for (int dim = 0; dim < getDimension(); dim++) {
				k5[dim] = myStep * getFunction().compute(dim, data);
			}
			// K6
			for (int dim = 0; dim < getDimension(); dim++) {
				data[dim] = data[dim] + B6 * k1[dim] + C6 * k2[dim] + D6 * k3[dim] + E6 * k4[dim] + F6 * k5[dim];
			}
			for (int dim = 0; dim < getDimension(); dim++) {
				k6[dim] = myStep * getFunction().compute(dim, data);
			}
			// maximum error
			for (int dim = 0; dim < getDimension(); dim++) {
				localError[dim] = Math.abs(R1 * k1[dim] + R3 * k3[dim] + R4 * k4[dim] + R5 * k5[dim] + R6 * k6[dim]);
				if (localError[dim] > error) {
					error = localError[dim];
				}
			}
			// reset data
			data = previous.toArray();
			// check error
			if (!hasMaxAbsDivergency() || error < getMaxAbsDivergency()) {
				// compute result
				for (int dim = 0; dim < getDimension(); dim++) {
					data[dim] = data[dim] + N1 * k1[dim] + N3 * k3[dim] + N4 * k4[dim] + N5 * k5[dim];
				}
				// check relative error
				if (error > 0 && hasMaxRelDivergency()) {
					double relError = 0;
					for (int dim = 0; dim < getDimension(); dim++) {
						double re = Math.abs(localError[dim] / data[dim]);
						if (re > relError) {
							relError = re;
						}
					}
					if (relError > getMaxRelDivergency()) {
						myStep /= 2;
						if(myStep < MINIMUM_TIME_STEP) {
							return new org.sybila.ode.cpu.Simulation(getDimension(), points.toArray(new Point[points.size()]), SimulationStatus.PRECISION);
						}
						continue;
					}

				}
				// save result
				currentTime += myStep;
				previous = new Point(getDimension(), currentTime, data);
				if (currentTime >= (position + 1) * timeStep) {
					position++;
					points.add(previous);
				}
				if (error < getMinAbsDivergency()) {
					myStep *= 2;
					if (myStep > timeStep) myStep = timeStep;
				}
			} else {
				myStep /= 2;
				if(myStep < MINIMUM_TIME_STEP) {
					return new org.sybila.ode.cpu.Simulation(getDimension(), points.toArray(new Point[points.size()]), SimulationStatus.PRECISION);
				}
			}

			if (currentTime >= targetTime) {
				break;
			}
			if (position >= getMaxSimulationSize()) {
				break;
			}
		}
		if (currentTime < targetTime) {
			return new org.sybila.ode.cpu.Simulation(getDimension(), points.toArray(new Point[points.size()]), SimulationStatus.TIMEOUT);
		}
		else {
			return new org.sybila.ode.cpu.Simulation(getDimension(), points.toArray(new Point[points.size()]), SimulationStatus.OK);
		}
	}
}
