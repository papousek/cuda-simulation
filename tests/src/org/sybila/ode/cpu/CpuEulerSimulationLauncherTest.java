package org.sybila.ode.cpu;

import org.sybila.ode.AbstractSimulationLauncherTest;
import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.SimulationLauncher;

public class CpuEulerSimulationLauncherTest extends AbstractSimulationLauncherTest
{

	@Override
	protected SimulationLauncher createLauncher(int dimension, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		return new CpuEulerSimulationLauncher(dimension, numberOfSimulations, maxSimulationSize, function);
	}


}
