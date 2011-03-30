package org.sybila.ode;

import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import static org.junit.Assert.*;

public abstract class AbstractSimulationLauncherTest
{

	private static final int DIMENSION = 10;

	private static final int NUMBER_OF_SIMULATIONS = 100;

	private static final int SIMULATION_LENGTH = 1000;

    private static Map<String, MultiAffineFunction> functions = new HashMap<String, MultiAffineFunction>();

	private static float[] vectors;

//	@Test
	public void valuesTest() {
        SimulationLauncher launcher = createLauncher(DIMENSION, NUMBER_OF_SIMULATIONS, SIMULATION_LENGTH, getFunctionForValuesTest());
		if (launcher == null) {
			return;
		}
		SimulationResult result = launcher.launch(0, (float) 0.25, 50, getVectors(), NUMBER_OF_SIMULATIONS);
		Simulation[] simulations = new Simulation[] {
			result.getSimulation(0),
			result.getSimulation(result.getNumberOfSimulations() / 2 + 1),
			result.getSimulation(result.getNumberOfSimulations() - 1),
		};
		for (int sim=0; sim<simulations.length; sim++) {
			assertTrue(simulations[sim].getLength() > 0);
			assertEquals(simulations[sim].getPoint(0).getValue(0), 0.865897, 0.0001);
			assertEquals(simulations[sim].getPoint(0).getTime(), 0.25, 0.0001);
		}
	}

    @Test
    public void simpleTest() {
        SimulationLauncher launcher = createLauncher(DIMENSION, NUMBER_OF_SIMULATIONS, SIMULATION_LENGTH, getFunctionForSimpleTest());
		if (launcher == null) {
			return;
		}
        SimulationResult result = launcher.launch(0, 1, SIMULATION_LENGTH, getVectors(), NUMBER_OF_SIMULATIONS);
		assertEquals(NUMBER_OF_SIMULATIONS, result.getNumberOfSimulations());
		Simulation[] simulations = new Simulation[] {
			result.getSimulation(0),
			result.getSimulation(1),
			result.getSimulation(result.getNumberOfSimulations() / 2 + 1),
			result.getSimulation(result.getNumberOfSimulations() - 1),
		};
		for (int sim=0; sim<simulations.length; sim++) {
			assertTrue(simulations[sim].getLength() >= Math.min(1000, SIMULATION_LENGTH));
			for(int i=0; i<Math.min(simulations[sim].getLength(), 10); i++) {
//				System.out.println(simulations[sim].getPoint(i));
				assertEquals("Time assertion in simulation [" + sim + "] step [" + i + "] failed.", (float) (1 * i + 1), simulations[sim].getPoint(i).getTime(), (float) 1);
			}
//			System.out.println();
		}
	
    }

	private static MultiAffineFunction getFunctionForSimpleTest() {
		if (functions.get("simpleTest") == null) {
			float[] coefficients = new float[DIMENSION];
			int[] coefficientIndexes = new int[DIMENSION + 1];
			int[] factors = new int[DIMENSION];
			int[] factorIndexes = new int[DIMENSION + 1];

			for (int i = 0; i < coefficients.length; i++) {
				coefficients[i] = (float) 0.005;
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

			functions.put("simpleTest", new MultiAffineFunction(coefficients, coefficientIndexes, factors, factorIndexes));
		}
		return functions.get("simpleTest");
	}

	private static MultiAffineFunction getFunctionForValuesTest() {
		if (functions.get("valuesTest") == null) {
			float[] coefficients = new float[2 * DIMENSION];
			int[] coefficientIndexes = new int[DIMENSION + 1];
			int[] factors = new int[2 * DIMENSION];
			int[] factorIndexes = new int[2 * DIMENSION + 1];
			for (int i=0; i < coefficients.length; i++) {
				coefficients[i] = (float) 1;
//				System.out.print(coefficients[i] + " ");
			}
//			System.out.println();
			for(int i=0; i < coefficientIndexes.length; i++) {
				coefficientIndexes[i] = i*2;
//				System.out.print(coefficientIndexes[i] + " ");
			}
//			System.out.println();
			for(int i=0; i < factors.length; i++) {
				factors[i] = i / 2;
//				System.out.print(factors[i] + " ");
			}
//			System.out.println();
			for(int i=0; i < factorIndexes.length; i++) {
				factorIndexes[i] = (i/2) * 2;
//				System.out.print(factorIndexes[i] + " ");
			}
//			System.out.println();
			functions.put("valuesTest", new MultiAffineFunction(coefficients, coefficientIndexes, factors, factorIndexes));
		}
		return functions.get("valuesTest");
	}

	private static float[] getVectors() {
		if (vectors == null) {
			vectors = new float[NUMBER_OF_SIMULATIONS * DIMENSION];
			for(int i=0; i <vectors.length; i++) {
				vectors[i] = (float) 0.5;
			}
		}
		return vectors;
	}


	abstract protected SimulationLauncher createLauncher(int dimension, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function);

}
