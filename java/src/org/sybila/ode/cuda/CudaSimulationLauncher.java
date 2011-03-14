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

	public CudaSimulationLauncher (int vectorSize, int numberOfSimulations, int maxNumberOfExecutedSteps, MultiAffineFunction function) {
		workspace = new SimulationWorkspace(vectorSize, numberOfSimulations, maxNumberOfExecutedSteps, function);
	}

	public SimulationResult launch(
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
