package org.sybila.ode;

import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public abstract class AbstractSimulationLauncherTest
{

    private MultiAffineFunction function;

    @Before
    public void setUp() {
        function = new MultiAffineFunction(
                new float[]{(float) 1.0, (float) 1.0, (float) 1.0},
                new int[]{0, 1, 2, 3},
                new int[]{0, 0, 0},
                new int[]{0, 1, 2, 3});
    }

    @Test
    public void simpleTest() {
        SimulationLauncher launcher = createLauncher(1, 1, 10, function);
        SimulationResult result = launcher.launch(0, (float) 0.1, 1, new float[]{(float) 1.0}, (float) 0.1, 100000);
        float[] values = new float[] {
            (float) 1.1051708,
            (float) 1.2214026,
            (float) 1.3498586,
            (float) 1.4918245,
            (float) 1.648721,
            (float) 1.8221185,
            (float) 2.0137525,
            (float) 2.2255404,
            (float) 2.4596028,
            (float) 2.7182815
        };
        assertEquals(1, result.getNumberOfSimulations());
		assertEquals(10, result.getSimulation(0).getLength());
        for(int i=0; i<result.getSimulation(0).getLength(); i++) {
//			System.out.println(result.getSimulation(0).getPoint(i));
            assertEquals(values[i], result.getSimulation(0).getPoint(i).getValue(0), (float) 0.1);
            assertEquals((float) (0.1 * i + 0.1), result.getSimulation(0).getPoint(i).getTime(), (float) 0.1);
        }
    }

	abstract protected SimulationLauncher createLauncher(int dimension, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function);

}
