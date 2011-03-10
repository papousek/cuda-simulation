package org.sybila.ode.cuda;

import java.io.File;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.utils.KernelLauncher;

public class CudaSimulationLauncher {

	private static final int BLOCK_SIZE = 32;

	private static final String KERNEL_FILE = "c" + File.separator + "build" + File.separator + "num_sim_kernel.cubin";

	private static final String KERNEL_NAME = "rkf45_kernel";

	private MultiAffineFunction function;

	public CudaSimulationLauncher (MultiAffineFunction function) {
		if (function == null) {
			throw new NullPointerException("The parameter [function] is null.");
		}
		this.function = function;
	}

	public NumericalSimulationResult launch(
		float	time,
		float	timeStep,
		float	targetTime,
		float[] vectors,
		int		numberOfVectors,
		int		vectorSize,
		float	absDivergency,
		int		maxNumberOfExecutedSteps,
		int		simulationSize
	) {
		// init data for device
		CUdeviceptr vectorsDevice		= new CUdeviceptr();
		CUdeviceptr resultPointsDevice	= new CUdeviceptr();
		CUdeviceptr resultTimesDevice	= new CUdeviceptr();
		CUdeviceptr returnCodesDevice	= new CUdeviceptr();
		CUdeviceptr executedStepsDevice	= new CUdeviceptr();

		JCuda.cudaMalloc(vectorsDevice, vectors.length * Sizeof.FLOAT);
		JCuda.cudaMalloc(resultPointsDevice, (simulationSize + 1) * vectorSize * Sizeof.FLOAT);
		JCuda.cudaMalloc(resultTimesDevice, simulationSize * Sizeof.FLOAT);
		JCuda.cudaMalloc(returnCodesDevice, numberOfVectors * Sizeof.INT);
		JCuda.cudaMalloc(executedStepsDevice, numberOfVectors * Sizeof.INT);

		JCuda.cudaMemcpy(vectorsDevice, Pointer.to(vectors), vectors.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);

		// init function data for device
		CUdeviceptr functionCoefficientsDevice			= new CUdeviceptr();
		CUdeviceptr functionCoefficientIndexesDevice	= new CUdeviceptr();
		CUdeviceptr functionFactorsDevice				= new CUdeviceptr();
		CUdeviceptr functionFactorIndexesDevice			= new CUdeviceptr();

		JCuda.cudaMalloc(functionCoefficientsDevice, function.getCoefficients().length * Sizeof.FLOAT);
		JCuda.cudaMalloc(functionCoefficientIndexesDevice, function.getCoefficientIndexes().length * Sizeof.INT);
		JCuda.cudaMalloc(functionFactorsDevice, function.getFactors().length * Sizeof.INT);
		JCuda.cudaMalloc(functionFactorIndexesDevice, function.getFactorIndexes().length * Sizeof.INT);

		JCuda.cudaMemcpy(functionCoefficientsDevice, Pointer.to(function.getCoefficients()), function.getCoefficients().length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
		JCuda.cudaMemcpy(functionCoefficientIndexesDevice, Pointer.to(function.getCoefficientIndexes()), function.getCoefficientIndexes().length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
		JCuda.cudaMemcpy(functionFactorsDevice, Pointer.to(function.getFactors()), function.getFactors().length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);
		JCuda.cudaMemcpy(functionFactorIndexesDevice, Pointer.to(function.getFactorIndexes()), function.getFactorIndexes().length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToDevice);

		// run kernel
		KernelLauncher launcher = KernelLauncher.load(KERNEL_FILE, KERNEL_NAME);
		int gridDim = (int) Math.ceil(Math.sqrt(numberOfVectors / (BLOCK_SIZE * vectorSize))) + 1;
		launcher.setGridSize(gridDim, gridDim);
		launcher.setBlockSize(BLOCK_SIZE, vectorSize, 1);
		launcher.call(
			time,
			timeStep,
			targetTime,
			vectorsDevice,
			numberOfVectors,
			vectorSize,
			absDivergency,
			maxNumberOfExecutedSteps,
			simulationSize,
			functionCoefficientsDevice,
			functionCoefficientIndexesDevice,
			functionFactorsDevice,
			functionFactorIndexesDevice,
			resultPointsDevice,
			resultTimesDevice,
			executedStepsDevice,
			returnCodesDevice
		);
		
		// save result
		int[] numberOfExecutedSteps 	= new int[numberOfVectors];
		int[] returnCodes				= new int[numberOfVectors];
		float[] simulationTimes			= new float[numberOfVectors * maxNumberOfExecutedSteps];
		float[] simulationPoints		= new float[numberOfVectors * vectorSize * maxNumberOfExecutedSteps];

		JCuda.cudaMemcpy(Pointer.to(numberOfExecutedSteps), executedStepsDevice, numberOfVectors * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
		JCuda.cudaMemcpy(Pointer.to(returnCodes), returnCodesDevice, numberOfVectors * Sizeof.INT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
		JCuda.cudaMemcpy(Pointer.to(simulationTimes), resultTimesDevice, numberOfVectors * Sizeof.FLOAT * maxNumberOfExecutedSteps, cudaMemcpyKind.cudaMemcpyDeviceToHost);
		JCuda.cudaMemcpy(Pointer.to(simulationPoints), resultPointsDevice, numberOfVectors * maxNumberOfExecutedSteps * vectorSize * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);

		// clean memory
		JCuda.cudaFree(vectorsDevice);
		JCuda.cudaFree(returnCodesDevice);
		JCuda.cudaFree(resultTimesDevice);
		JCuda.cudaFree(resultPointsDevice);
		JCuda.cudaFree(executedStepsDevice);
		JCuda.cudaFree(functionCoefficientsDevice);
		JCuda.cudaFree(functionCoefficientIndexesDevice);
		JCuda.cudaFree(functionFactorsDevice);
		JCuda.cudaFree(functionFactorIndexesDevice);

		return new NumericalSimulationResult(numberOfVectors, vectorSize, numberOfExecutedSteps, returnCodes, simulationTimes, simulationPoints);
	}

}
