package org.sybila.ode;

import org.junit.Before;
import org.junit.Test;
import org.sybila.ode.cpu.CpuEulerSimulationLauncher;
import static org.junit.Assert.*;

public abstract class AbstractSimulationLauncherTest
{

	private static final int DIMENSION = 3;

	private static final int NUMBER_OF_SIMULATIONS = 100;

	private static final int SIMULATION_LENGTH = 10;

    private static MultiAffineFunction function;

	private static float[] vectors;

    @Test
    public void simpleTest() {
        SimulationLauncher launcher = createLauncher(DIMENSION, NUMBER_OF_SIMULATIONS, SIMULATION_LENGTH, getFunction());
        SimulationResult result = launcher.launch(0, 1, SIMULATION_LENGTH, getVectors(), NUMBER_OF_SIMULATIONS, (float) 0.001, (int) (SIMULATION_LENGTH / CpuEulerSimulationLauncher.TIME_STEP) * 10);
        float[] values = new float[] {
            (float) 2.7,
            (float) 7.6,
            (float) 20.6,
            (float) 56.0,
            (float) 153.0,
            (float) 414.6,
            (float) 1126.9,
            (float) 3064.6,
            (float) 8325.6,
            (float) 22621.3
        };
        assertEquals(NUMBER_OF_SIMULATIONS, result.getNumberOfSimulations());
		assertEquals(SIMULATION_LENGTH, result.getSimulation(0).getLength());
        for(int i=0; i<result.getSimulation(50).getLength(); i++) {
//			System.out.println(result.getSimulation(0).getPoint(i));
//            assertEquals(values[i], result.getSimulation(0).getPoint(i).getValue(0), (float) values[i] * 0.1);
			assertEquals((float) (1 * i + 1), result.getSimulation(0).getPoint(i).getTime(), (float) 1);
        }
    }

	private static MultiAffineFunction getFunction() {
		if (function == null) {
			float[] coefficients = new float[DIMENSION];
			int[] coefficientIndexes = new int[DIMENSION + 1];
			int[] factors = new int[DIMENSION];
			int[] factorIndexes = new int[DIMENSION + 1];

			for (int i = 0; i < coefficients.length; i++) {
				coefficients[i] = (float) 1;
			}

			for (int i = 0; i < coefficientIndexes.length; i++) {
				coefficientIndexes[i] = i;
			}

			for (int i = 0; i < factors.length; i++) {
				factors[i] = i;
			}

			for (int i = 0; i < factorIndexes.length; i++) {
				factorIndexes[i] = i;
			}

			function = new MultiAffineFunction(coefficients, coefficientIndexes, factors, factorIndexes);
		}
		return function;
	}

	private static float[] getVectors() {
		if (vectors == null) {
			vectors = new float[NUMBER_OF_SIMULATIONS * DIMENSION];
			for(int i=0; i <vectors.length; i++) {
				vectors[i] = 1;
			}
		}
		return vectors;
	}


	abstract protected SimulationLauncher createLauncher(int dimension, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function);

}
