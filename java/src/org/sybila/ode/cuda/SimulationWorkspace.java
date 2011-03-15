package org.sybila.ode.cuda;

import jcuda.CudaException;
import org.sybila.ode.MultiAffineFunction;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.cudaStream_t;

public class SimulationWorkspace {

	private MultiAffineFunction function;

	private boolean initialized = false;

	private int maxSimulationSize;

	private int maxNumberOfSimulations;

	private int dimension;

	private cudaStream_t stream;

	private CUdeviceptr functionCoefficients;

	private CUdeviceptr functionCoefficientIndexes;

	private CUdeviceptr functionFactors;

	private CUdeviceptr functionFactorIndexes;

	private CUdeviceptr vectors;

	private CUdeviceptr resultPoints;

	private CUdeviceptr resultTimes;

	private CUdeviceptr returnCodes;

	private CUdeviceptr executedSteps;

	public SimulationWorkspace(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function, cudaStream_t stream) {
		this(dimension, maxNumberOfSimulations, maxSimulationSize, function);
		if (stream == null) {
			throw new NullPointerException("The parameter [stream] is NULL.");
		}
		this.stream = stream;
	}

	public SimulationWorkspace(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		if (dimension <= 0) {
			throw new IllegalArgumentException("The parameter [dimension] has to be a positive number.");
		}
		if (maxNumberOfSimulations <= 0) {
			throw new IllegalArgumentException("The parameter [maxNumberOfSimulations] has to be a positive number.");
		}
		if (maxSimulationSize <= 0) {
			throw new IllegalArgumentException("The parameter [maxSimulationSize] has to be a positive number.");
		}
		if (function == null) {
			throw new NullPointerException("The parameter [function] is NULL.");
		}
		this.dimension				= dimension;
		this.maxSimulationSize		= maxSimulationSize;
		this.maxNumberOfSimulations	= maxNumberOfSimulations;
		this.function				= function;
	}

	public Pointer getDeviceExecutedSteps() {
		initPointers();
		return executedSteps;
	}

	public Pointer getDeviceFunctionCoefficients() {
		initPointers();
		return functionCoefficients;
	}

	public Pointer getDeviceFunctionCoefficientIndexes() {
		initPointers();
		return functionCoefficientIndexes;
	}

	public Pointer getDeviceFunctionFactors() {
		initPointers();
		return functionFactors;
	}

	public Pointer getDeviceFunctionFactorIndexes() {
		initPointers();
		return functionFactorIndexes;
	}

	public Pointer getDeviceResultPoints() {
		initPointers();
		return resultPoints;
	}

	public Pointer getDeviceResultTimes() {
		initPointers();
		return resultTimes;
	}

	public Pointer getDeviceReturnCodes() {
		initPointers();
		return returnCodes;
	}

	public Pointer getDeviceVectors() {
		initPointers();
		return vectors;
	}

	public MultiAffineFunction getFunction() {
		return function;
	}

	public int getMaxNumberOfSimulations() {
		return maxNumberOfSimulations;
	}

	public int getMaxSimulationSize() {
		return maxSimulationSize;
	}

	public SimulationResult getResult(int numberOfSimulations) {
		if (numberOfSimulations <= 0 || numberOfSimulations > getMaxNumberOfSimulations()) {
			throw new IllegalArgumentException("The number of simulations [" + numberOfSimulations + "] is out of the range [1, " + getMaxNumberOfSimulations() + "].");
		}
		initPointers();
		int[] numberOfExecutedStepsHost	= new int[numberOfSimulations];
		int[] returnCodesHost			= new int[numberOfSimulations];
		float[] simulationTimesHost		= new float[numberOfSimulations * maxSimulationSize];
		float[] simulationPointsHost	= new float[numberOfSimulations * dimension * maxSimulationSize];
		
		copyDeviceToHost(Pointer.to(numberOfExecutedStepsHost), executedSteps, numberOfSimulations * Sizeof.INT);
		copyDeviceToHost(Pointer.to(returnCodesHost), returnCodes, numberOfSimulations * Sizeof.INT);
		copyDeviceToHost(Pointer.to(simulationTimesHost), resultTimes, simulationTimesHost.length * Sizeof.FLOAT);
		copyDeviceToHost(Pointer.to(simulationPointsHost), resultPoints, numberOfSimulations * maxSimulationSize * dimension * Sizeof.FLOAT);

		return new SimulationResult(numberOfSimulations, dimension, numberOfExecutedStepsHost, returnCodesHost, simulationTimesHost, simulationPointsHost);
	}

	public int getDimension() {
		return dimension;
	}

	public void bindVectors(float[] vectors) {
		if (vectors == null) {
			throw new NullPointerException("The parameter [vectors] is NULL.");
		}
		if (vectors.length > dimension * maxNumberOfSimulations) {
			throw new IllegalArgumentException("The size of array [vector] doesn't correspond to the product of number of simulations and vector size.");
		}
		initPointers();
		copyHostToDevice(this.vectors, Pointer.to(vectors), vectors.length * Sizeof.FLOAT);
	}

	public void destroy() {
		JCuda.cudaFree(executedSteps);
		JCuda.cudaFree(functionCoefficientIndexes);
		JCuda.cudaFree(functionCoefficients);
		JCuda.cudaFree(functionFactorIndexes);
		JCuda.cudaFree(functionFactors);
		JCuda.cudaFree(resultPoints);
		JCuda.cudaFree(resultTimes);
		JCuda.cudaFree(returnCodes);
		JCuda.cudaFree(vectors);
	}

	private void initPointers() {
		if (initialized) {
			return;
		}
		initialized = true;
		vectors			= new CUdeviceptr();
		resultPoints	= new CUdeviceptr();
		resultTimes		= new CUdeviceptr();
		returnCodes		= new CUdeviceptr();
		executedSteps	= new CUdeviceptr();

		functionCoefficients		= new CUdeviceptr();
		functionCoefficientIndexes	= new CUdeviceptr();
		functionFactors				= new CUdeviceptr();
		functionFactorIndexes		= new CUdeviceptr();

		JCuda.cudaMalloc(vectors, maxNumberOfSimulations * dimension * Sizeof.FLOAT);
		JCuda.cudaMalloc(resultPoints, (maxSimulationSize + 1) * maxNumberOfSimulations * dimension * Sizeof.FLOAT);
		JCuda.cudaMalloc(resultTimes, maxNumberOfSimulations * maxSimulationSize * Sizeof.FLOAT);
		JCuda.cudaMalloc(returnCodes, maxNumberOfSimulations * Sizeof.INT);
		JCuda.cudaMalloc(executedSteps, maxNumberOfSimulations * Sizeof.INT);

		JCuda.cudaMalloc(functionCoefficients, function.getCoefficients().length * Sizeof.FLOAT);
		JCuda.cudaMalloc(functionCoefficientIndexes, function.getCoefficientIndexes().length * Sizeof.INT);
		JCuda.cudaMalloc(functionFactors, function.getFactors().length * Sizeof.INT);
		JCuda.cudaMalloc(functionFactorIndexes, function.getFactorIndexes().length * Sizeof.INT);

		copyHostToDevice(functionCoefficients, Pointer.to(function.getCoefficients()), function.getCoefficients().length * Sizeof.FLOAT);
		copyHostToDevice(functionCoefficientIndexes, Pointer.to(function.getCoefficientIndexes()), function.getCoefficientIndexes().length * Sizeof.INT);
		copyHostToDevice(functionFactors, Pointer.to(function.getFactors()), function.getFactors().length * Sizeof.INT);
		copyHostToDevice(functionFactorIndexes, Pointer.to(function.getFactorIndexes()), function.getFactorIndexes().length * Sizeof.INT);
	}

	private void copyDeviceToHost(Pointer hostPointer, Pointer devicePointer, int size) {
		if (stream == null) {
			JCuda.cudaMemcpy(hostPointer, devicePointer, size, cudaMemcpyKind.cudaMemcpyDeviceToHost);
		}
		else {
			JCuda.cudaMemcpyAsync(hostPointer, devicePointer, size, cudaMemcpyKind.cudaMemcpyDeviceToHost, stream);
		}
		int error = JCuda.cudaGetLastError();
		if (error != cudaError.cudaSuccess) {
			throw new CudaException(JCuda.cudaGetErrorString(error));
		}
	}

	private void copyHostToDevice(Pointer devicePointer, Pointer hostPointer, int size) {
		if (stream == null) {
			JCuda.cudaMemcpy(devicePointer, hostPointer, size, cudaMemcpyKind.cudaMemcpyHostToDevice);
		}
		else {
			JCuda.cudaMemcpyAsync(devicePointer, hostPointer, size, cudaMemcpyKind.cudaMemcpyHostToDevice, stream);
		}
		int error = JCuda.cudaGetLastError();
		if (error != cudaError.cudaSuccess) {
			throw new CudaException(JCuda.cudaGetErrorString(error));
		}
	}

	private void copyHostToDeviceConstant(String deviceConstant, Pointer hostPointer, int size, int offset) {
		if (stream == null) {
			JCuda.cudaMemcpyToSymbol(deviceConstant, hostPointer, size, offset, cudaMemcpyKind.cudaMemcpyHostToDevice);
		}
		else {
			JCuda.cudaMemcpyToSymbolAsync(deviceConstant, hostPointer, size, offset, cudaMemcpyKind.cudaMemcpyHostToDevice, stream);
		}
		int error = JCuda.cudaGetLastError();
		if (error != cudaError.cudaSuccess) {
			throw new CudaException(JCuda.cudaGetErrorString(error));
		}
	}

	private static void printArray(int[] array) {
		for (int i=0; i<array.length; i++) {
			System.out.println(array[i]);
		}
	}

	private static void printArray(float[] array) {
		for (int i=0; i<array.length; i++) {
			System.out.println(array[i]);
		}
	}

}