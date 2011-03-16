package org.sybila.ode.cuda;

import java.io.File;
import org.sybila.ode.MultiAffineFunction;

public class CudaRkf45SimulationLauncher extends AbstractCudaSimulationLauncher
{

	private static final String KERNEL_FILE = "c" + File.separator + "build" + File.separator + "num_sim_kernel.cubin";

	private static final String KERNEL_NAME = "rkf45_kernel";

	public CudaRkf45SimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function);
	}

	@Override
	protected String getKernelFile() {
		return KERNEL_FILE;
	}

	@Override
	protected String getKernelName() {
		return KERNEL_NAME;
	}


}
