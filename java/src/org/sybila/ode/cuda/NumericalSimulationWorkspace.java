package org.sybila.ode.cuda;

import org.sybila.ode.MultiAffineFunction;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.cudaStream_t;

public class NumericalSimulationWorkspace {

	private MultiAffineFunction function;
	private boolean initialized = false;
	private int maxSimulationSize;
	private int numberOfSimulations;
	private int vectorSize;
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

	public NumericalSimulationWorkspace(int vectorSize, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function, cudaStream_t stream) {
		this(vectorSize, numberOfSimulations, maxSimulationSize, function);
		if (stream == null) {
			throw new NullPointerException("The parameter [stream] is NULL.");
		}
		this.stream = stream;
	}

	public NumericalSimulationWorkspace(int vectorSize, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		if (vectorSize <= 0) {
			throw new IllegalArgumentException("The parameter [vectorSize] has to be a positive number.");
		}
		if (numberOfSimulations <= 0) {
			throw new IllegalArgumentException("The parameter [numberOfSimulations] has to be a positive number.");
		}
		if (maxSimulationSize <= 0) {
			throw new IllegalArgumentException("The parameter [maxSimulationSize] has to be a positive number.");
		}
		if (function == null) {
			throw new NullPointerException("The parameter [function] is NULL.");
		}
		this.vectorSize = vectorSize;
		this.maxSimulationSize = maxSimulationSize;
		this.numberOfSimulations = numberOfSimulations;
		this.function = function;
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

	public int getMaxSimulationSize() {
		return maxSimulationSize;
	}

	public int getNumberOfSimulations() {
		return numberOfSimulations;
	}

	public SimulationResult getResult() {
		initPointers();
		int[] numberOfExecutedStepsHost = new int[numberOfSimulations];
		int[] returnCodesHost = new int[numberOfSimulations];
		float[] simulationTimesHost = new float[numberOfSimulations * maxSimulationSize];
		float[] simulationPointsHost = new float[numberOfSimulations * vectorSize * maxSimulationSize];

		copyDeviceToHost(Pointer.to(numberOfExecutedStepsHost), executedSteps, numberOfSimulations * Sizeof.INT);
		copyDeviceToHost(Pointer.to(returnCodesHost), returnCodes, numberOfSimulations * Sizeof.INT);
		copyDeviceToHost(Pointer.to(simulationTimesHost), resultTimes, simulationTimesHost.length * Sizeof.FLOAT);
		copyDeviceToHost(Pointer.to(simulationPointsHost), resultPoints, numberOfSimulations * maxSimulationSize * vectorSize * Sizeof.FLOAT);

		return new SimulationResult(numberOfSimulations, vectorSize, numberOfExecutedStepsHost, returnCodesHost, simulationTimesHost, simulationPointsHost);
	}

	public int getVectorSize() {
		return vectorSize;
	}

	public void bindVectors(float[] vectors) {
		if (vectors == null) {
			throw new NullPointerException("The parameter [vectors] is NULL.");
		}
		if (vectors.length != vectorSize * numberOfSimulations) {
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
		vectors = new CUdeviceptr();
		resultPoints = new CUdeviceptr();
		resultTimes = new CUdeviceptr();
		returnCodes = new CUdeviceptr();
		executedSteps = new CUdeviceptr();

		functionCoefficients = new CUdeviceptr();
		functionCoefficientIndexes = new CUdeviceptr();
		functionFactors = new CUdeviceptr();
		functionFactorIndexes = new CUdeviceptr();

		JCuda.cudaMalloc(vectors, numberOfSimulations * vectorSize * Sizeof.FLOAT);
		JCuda.cudaMalloc(resultPoints, (maxSimulationSize + 1) * numberOfSimulations * vectorSize * Sizeof.FLOAT);
		JCuda.cudaMalloc(resultTimes, numberOfSimulations * maxSimulationSize * Sizeof.FLOAT);
		JCuda.cudaMalloc(returnCodes, numberOfSimulations * Sizeof.INT);
		JCuda.cudaMalloc(executedSteps, numberOfSimulations * Sizeof.INT);

		JCuda.cudaMalloc(functionCoefficients, function.getCoefficients().length * Sizeof.FLOAT);
		JCuda.cudaMalloc(functionCoefficientIndexes, function.getCoefficientIndexes().length * Sizeof.INT);
		JCuda.cudaMalloc(functionFactors, function.getFactors().length * Sizeof.INT);
		JCuda.cudaMalloc(functionFactorIndexes, function.getFactorIndexes().length * Sizeof.INT);

/*		copyHostToDevicConst("function_coefficients", Pointer.to(function.getCoefficients()), Sizeof.FLOAT * function.getCoefficients().length, 0);
		copyHostToDevicConst("function_coefficient_indexes", Pointer.to(function.getCoefficientIndexes()), Sizeof.INT * function.getCoefficientIndexes().length, 0);
		copyHostToDevicConst("function_factors", Pointer.to(function.getFactors()), Sizeof.INT * function.getFactors().length, 0);
		copyHostToDevicConst("function_factor_indexes", Pointer.to(function.getFactorIndexes()), Sizeof.INT * function.getFactorIndexes().length, 0);
*/	}

	private void copyDeviceToHost(Pointer hostPointer, Pointer devicePointer, int size) {
		if (stream == null) {
			JCuda.cudaMemcpy(hostPointer, devicePointer, size, cudaMemcpyKind.cudaMemcpyDeviceToHost);
		} else {
			JCuda.cudaMemcpyAsync(hostPointer, devicePointer, size, cudaMemcpyKind.cudaMemcpyDeviceToHost, stream);
		}
	}

	private void copyHostToDevice(Pointer devicePointer, Pointer hostPointer, int size) {
		if (stream == null) {
			JCuda.cudaMemcpy(devicePointer, hostPointer, size, cudaMemcpyKind.cudaMemcpyHostToDevice);
		} else {
			JCuda.cudaMemcpyAsync(devicePointer, hostPointer, size, cudaMemcpyKind.cudaMemcpyHostToDevice, stream);
		}
	}

	private void copyHostToDevicConst(String deviceConstant, Pointer hostPointer, int size, int offset) {
		if (stream == null) {
			JCuda.cudaMemcpyToSymbol(deviceConstant, hostPointer, size, offset, cudaMemcpyKind.cudaMemcpyHostToDevice);
		}
		else {
			JCuda.cudaMemcpyToSymbolAsync(deviceConstant, hostPointer, size, offset, cudaMemcpyKind.cudaMemcpyHostToDevice, stream);
		}
	}
}
