package org.sybila.ode.cuda;

import org.sybila.ode.AbstractSimulationLauncherTest;
import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.SimulationLauncher;

public class CudaEulerSimulationLauncherTest extends AbstractSimulationLauncherTest
{

	@Override
	protected SimulationLauncher createLauncher(int dimension, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		return new CudaEulerSimulationLauncher(dimension, numberOfSimulations, maxSimulationSize, function);
	}

}
