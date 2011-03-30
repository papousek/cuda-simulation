package org.sybila.ode.cpu;

import org.sybila.ode.AbstractSimulationLauncherTest;
import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.SimulationLauncher;

public class CpuRkf45SimulationLauncherTest extends AbstractSimulationLauncherTest
{

	@Override
	protected SimulationLauncher createLauncher(int dimension, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
//		return null;
		return new CpuRkf45SimulationLauncher(dimension, numberOfSimulations, maxSimulationSize, function);
	}


}
