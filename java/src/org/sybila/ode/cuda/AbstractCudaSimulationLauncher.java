package org.sybila.ode.cuda;

import jcuda.runtime.cudaStream_t;
import jcuda.utils.KernelLauncher;
import org.sybila.ode.AbstractSimulationLauncher;
import org.sybila.ode.MultiAffineFunction;

abstract public class AbstractCudaSimulationLauncher extends AbstractSimulationLauncher
{

	private static final int BLOCK_SIZE = 32;

	private SimulationWorkspace workspace;

	private cudaStream_t stream;

	public AbstractCudaSimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function);
	}

	public SimulationResult launch(
		float	time,
		float	timeStep,
		float	targetTime,
		float[] vectors,
		int		numberOfSimulations,
		float	absDivergency
	) {
		if (numberOfSimulations <= 0 || numberOfSimulations > getMaxNumberOfSimulations()) {
			throw new IllegalArgumentException("The number of simulations [" + numberOfSimulations + "] is out of the range [1, " + workspace.getMaxNumberOfSimulations() + "].");
		}

		// prepare data
		getWorkspace().bindVectors(vectors);

		// run kernel
		KernelLauncher launcher = KernelLauncher.load(getKernelFile(), getKernelName());
		int gridDim = (int) Math.ceil(Math.sqrt((float) numberOfSimulations / BLOCK_SIZE));
		launcher.setGridSize(gridDim, gridDim);
		launcher.setBlockSize(BLOCK_SIZE, workspace.getDimension(), 1);
		launcher.call(
			time,
			timeStep,
			targetTime,
			getWorkspace().getDeviceVectors(),
			numberOfSimulations,
			getWorkspace().getMaxNumberOfSimulations(),
			getWorkspace().getDimension(),
			absDivergency,
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

	protected void clean() {
		if (workspace == null) {
			workspace.destroy();
		}
	}

	protected cudaStream_t getStream() {
		if (stream == null) {
			stream = new cudaStream_t();
		}
		return stream;
	}

	protected SimulationWorkspace getWorkspace() {
		if (workspace == null) {
			workspace = new SimulationWorkspace(getDimension(), getMaxNumberOfSimulations(), getMaxSimulationSize(), getFunction());
		}
		return workspace;
	}

	abstract protected String getKernelFile();

	abstract protected String getKernelName();

}
