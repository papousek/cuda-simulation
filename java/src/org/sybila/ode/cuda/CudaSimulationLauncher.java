package org.sybila.ode.cuda;

import java.io.File;
import jcuda.utils.KernelLauncher;

public class CudaSimulationLauncher {

	private static final int BLOCK_SIZE = 32;

	private static final String KERNEL_FILE = "c" + File.separator + "build" + File.separator + "num_sim_kernel.cubin";

	private static final String KERNEL_NAME = "rkf45_kernel";

	private NumericalSimulationWorkspace workspace;

	public CudaSimulationLauncher (int vectorSize, int numberOfSimulations, int maxNumberOfExecutedSteps, MultiAffineFunction function) {
		workspace = new NumericalSimulationWorkspace(vectorSize, numberOfSimulations, maxNumberOfExecutedSteps, function);
	}

	public NumericalSimulationResult launch(
		float	time,
		float	timeStep,
		float	targetTime,
		float[] vectors,
		float	absDivergency,
		int		maxNumberOfExecutedSteps
	) {

		// prepare data
		workspace.bindVectors(vectors);

		// run kernel
		KernelLauncher launcher = KernelLauncher.load(KERNEL_FILE, KERNEL_NAME);
		int gridDim = (int) Math.ceil(Math.sqrt( workspace.getNumberOfSimulations() / (BLOCK_SIZE * workspace.getVectorSize()))) + 1;
		launcher.setGridSize(gridDim, gridDim);
		launcher.setBlockSize(BLOCK_SIZE, workspace.getVectorSize(), 1);
		launcher.call(
			time,
			timeStep,
			targetTime,
			workspace.getDeviceVectors(),
			workspace.getNumberOfSimulations(),
			workspace.getVectorSize(),
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
		return workspace.getResult();
	}

}
