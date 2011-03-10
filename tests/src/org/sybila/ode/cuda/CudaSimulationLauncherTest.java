/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package org.sybila.ode.cuda;

import org.junit.Before;
import org.junit.Test;

public class CudaSimulationLauncherTest {

	private MultiAffineFunction function;

    @Before
    public void setUp() {
		function = new MultiAffineFunction(
			new float[] {(float) 1.0, (float) 1.0, (float) 1.0},
			new int[] {0, 1, 2, 3 },
			new int[] {0, 0, 0 },
			new int[] {0, 1, 2, 3}
		);
    }

	@Test
	public void simpleTest() {
		CudaSimulationLauncher launcher = new CudaSimulationLauncher(1, 1, 10, function);
		NumericalSimulationResult result = launcher.launch(0, (float) 0.1, 1, new float[] {(float) 1.0 }, (float) 0.1, 100);
		for (int i=0; i<result.getNumberOfExecutedSteps(0); i++) {
			System.out.println(result.getPoint(0, i));
		}
	}

}