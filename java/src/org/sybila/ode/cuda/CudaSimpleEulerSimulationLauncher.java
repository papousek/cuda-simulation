package org.sybila.ode.cuda;

import java.io.File;
import jcuda.utils.KernelLauncher;
import org.sybila.ode.MultiAffineFunction;

public class CudaSimpleEulerSimulationLauncher extends AbstractCudaSimulationLauncher
{

	private static final String KERNEL_FILE = "c" + File.separator + "build" + File.separator + "num_sim_kernel.cubin";

	private static final String KERNEL_NAME = "euler_simple_kernel";

	public CudaSimpleEulerSimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function);
	}

	@Override
	public SimulationResult launch(
		float	time,
		float	timeStep,
		float	targetTime,
		float[] vectors,
		int		numberOfSimulations
	) {
		if (numberOfSimulations <= 0 || numberOfSimulations > getMaxNumberOfSimulations()) {
			throw new IllegalArgumentException("The number of simulations [" + numberOfSimulations + "] is out of the range [1, " + getWorkspace().getMaxNumberOfSimulations() + "].");
		}

		// prepare data
		getWorkspace().bindVectors(vectors);

		// run kernel
		KernelLauncher launcher = KernelLauncher.load(getKernelFile(), getKernelName());
		int gridDim = (int) Math.ceil(Math.sqrt((float) numberOfSimulations / (BLOCK_SIZE * BLOCK_SIZE/2)));
		launcher.setGridSize(gridDim, gridDim);
		launcher.setBlockSize(BLOCK_SIZE, BLOCK_SIZE/2, 1);
		launcher.call(
			time,
			timeStep,
			targetTime,
			getWorkspace().getDeviceVectors(),
			numberOfSimulations,
			getWorkspace().getMaxNumberOfSimulations(),
			getWorkspace().getDimension(),
			getMinAbsDivergency(),
			getMaxAbsDivergency(),
			getMinRelDivergency(),
			getMaxRelDivergency(),
			MAX_NUMBER_OF_EXECUTED_STEPS,
			getWorkspace().getMaxSimulationSize(),
			getWorkspace().getDeviceFunctionCoefficients(),
			getWorkspace().getDeviceFunctionCoefficientIndexes(),
			getWorkspace().getDeviceFunctionFactors(),
			getWorkspace().getDeviceFunctionFactorIndexes(),
			getWorkspace().getDeviceResultPoints(),
			getWorkspace().getDeviceResultTimes(),
			getWorkspace().getDeviceExecutedSteps(),
			getWorkspace().getDeviceReturnCodes()
		);

		// get result
		return getWorkspace().getResult(numberOfSimulations);
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
