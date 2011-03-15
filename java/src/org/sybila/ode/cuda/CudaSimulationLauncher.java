package org.sybila.ode.cuda;

import org.sybila.ode.MultiAffineFunction;
import java.io.File;
import jcuda.utils.KernelLauncher;
import org.sybila.ode.SimulationLauncher;
import org.sybila.ode.SimulationResult;

public class CudaSimulationLauncher implements SimulationLauncher
{

	private static final int BLOCK_SIZE = 32;

	private static final String KERNEL_FILE = "c" + File.separator + "build" + File.separator + "num_sim_kernel.cubin";

	private static final String KERNEL_NAME = "rkf45_kernel";

	private SimulationWorkspace workspace;

	public CudaSimulationLauncher (int vectorSize, int maxNumberOfSimulations, int maxNumberOfExecutedSteps, MultiAffineFunction function) {
		workspace = new SimulationWorkspace(vectorSize, maxNumberOfSimulations, maxNumberOfExecutedSteps, function);
	}

	public SimulationResult launch(
		float	time,
		float	timeStep,
		float	targetTime,
		float[] vectors,
		int		numberOfSimulations,
		float	absDivergency,
		int		maxNumberOfExecutedSteps
	) {

		if (numberOfSimulations <= 0 || numberOfSimulations > workspace.getMaxNumberOfSimulations()) {
			throw new IllegalArgumentException("The number of simulations [" + numberOfSimulations + "] is out of the range [1, " + workspace.getMaxNumberOfSimulations() + "].");
		}

		// prepare data
		workspace.bindVectors(vectors);

		// run kernel
		KernelLauncher launcher = KernelLauncher.load(KERNEL_FILE, KERNEL_NAME);
		int gridDim = (int) Math.ceil(Math.sqrt( workspace.getMaxNumberOfSimulations() / (BLOCK_SIZE * workspace.getDimension()))) + 1;
		launcher.setGridSize(gridDim, gridDim);
		launcher.setBlockSize(BLOCK_SIZE, workspace.getDimension(), 1);
		launcher.call(
			time,
			timeStep,
			targetTime,
			workspace.getDeviceVectors(),
			numberOfSimulations,
			workspace.getMaxNumberOfSimulations(),
			workspace.getDimension(),
			absDivergency,
			maxNumberOfExecutedSteps,
			workspace.getMaxSimulationSize(),
			workspace.getDeviceFunctionCoefficients(),
			workspace.getDeviceFunctionCoefficientIndexes(),
			workspace.getDeviceFunctionFactors(),
			workspace.getDeviceFunctionFactorIndexes(),
			workspace.getDeviceResultPoints(),
			workspace.getDeviceResultTimes(),
			workspace.getDeviceExecutedSteps(),
			workspace.getDeviceReturnCodes()
		);
		

		// get result
		return workspace.getResult(numberOfSimulations);
	}

	public void destroy() {
		if (workspace == null) {
			workspace.destroy();
		}
	}

}
