package org.sybila.ode.benchmark;

import jcuda.Sizeof;
import jcuda.runtime.cudaDeviceProp;
import org.sybila.ode.EquationSystem;
import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.SimulationLauncher;
import org.sybila.ode.cpu.CpuEulerSimulationLauncher;
import org.sybila.ode.cpu.CpuRkf45SimulationLauncher;
import org.sybila.ode.cuda.CudaEulerSimulationLauncher;
import org.sybila.ode.cuda.CudaRkf45SimulationLauncher;
import org.sybila.ode.cuda.CudaSimpleEulerSimulationLauncher;

public class Main {

	private static final int DIMENSION = 10;
	private static final int MAX_NUMBER_OF_SIMULATIONS = 30000;
	private static final int MIN_NUMBER_OF_SIMULATIONS = 1000;
	private static final int NUMBER_OF_SIMULATIONS_STEP = 1000;
	private static final int SIMULATION_LENGTH = 1000;

	public static void main(String[] args) {
		EquationSystem system = SimulationBenchmark.createLinearFunction(DIMENSION);
		header(system);
		MultiAffineFunction function = system.getFunction();
		SimulationLauncher[] launchers = new SimulationLauncher[]{
			new CpuEulerSimulationLauncher(DIMENSION, MAX_NUMBER_OF_SIMULATIONS, SIMULATION_LENGTH, function),
			new CpuRkf45SimulationLauncher(DIMENSION, MAX_NUMBER_OF_SIMULATIONS, SIMULATION_LENGTH, function),
//			new CudaRkf45SimulationLauncher(DIMENSION, MAX_NUMBER_OF_SIMULATIONS,SIMULATION_LENGTH, function),
//			new CudaEulerSimulationLauncher(DIMENSION, MAX_NUMBER_OF_SIMULATIONS,SIMULATION_LENGTH, function),
//			new CudaSimpleEulerSimulationLauncher(DIMENSION, MAX_NUMBER_OF_SIMULATIONS,SIMULATION_LENGTH, function),
		};

		for (int numberOfSimulations = MIN_NUMBER_OF_SIMULATIONS; numberOfSimulations <= MAX_NUMBER_OF_SIMULATIONS; numberOfSimulations += NUMBER_OF_SIMULATIONS_STEP) {
			line();
			System.out.println("  NUMBER OF SIMULATIONS: " + numberOfSimulations);
			line();
			SimulationBenchmark benchmark = new SimulationBenchmark(DIMENSION, numberOfSimulations, SIMULATION_LENGTH);
			for (SimulationLauncher launcher : launchers) {
				result(launcher, benchmark.bench(launcher));
				System.gc();
			}
		}
	}

	private static void header(EquationSystem system) {
		line();
		System.out.println("  DIMENSION:                     " + DIMENSION);
		System.out.println("  SIMULATION LENGTH:             " + SIMULATION_LENGTH);
		System.out.println("  MINIMUM NUMBER OF SIMULATIONS: " + MIN_NUMBER_OF_SIMULATIONS);
		System.out.println("  MAXIMUM NUMBER OF SIMULATIONS: " + MAX_NUMBER_OF_SIMULATIONS);
		System.out.println("  NUMBER OF SIMULATIONS STEP:    " + NUMBER_OF_SIMULATIONS_STEP);
		cudaDeviceProp properties = new cudaDeviceProp();
		jcuda.runtime.JCuda.cudaGetDeviceProperties(properties, 0);
		System.out.println("  SIZE OF GLOBAL MEMORY:         " + properties.totalGlobalMem / (1000 * 1000 * Sizeof.FLOAT) + " Mfloats");
		System.out.println("  MAXIMUM MEMORY PITCH:          " + properties.memPitch / (1000 * 1000 * Sizeof.FLOAT) + " Mfloats");
		System.out.println("  KERNEL TIMEOUT:                " + (properties.kernelExecTimeoutEnabled > 1 ? "YES" : "NO"));
		System.out.println("  MAX THREADS PER BLOCK:         " + properties.maxThreadsPerBlock);
		System.out.println("  MAX THREADS DIM:               [" + properties.maxThreadsDim[0] + ", " + properties.maxThreadsDim[1] + ", " + properties.maxThreadsDim[2] + "]");
		System.out.println("  MAX GRID SIZE:                 [" + properties.maxGridSize[0] + ", " + properties.maxGridSize[1] + ", " + properties.maxGridSize[2] + "]");
		System.out.println("  HEAP SIZE:                     " + Runtime.getRuntime().totalMemory() / (1024 * 1024) + " MB");
		line();
		System.out.println("  EQUATION SYSTEM: ");
		System.out.println(system);
		line();
	}

	private static void line() {
		System.out.println("--------------------------------------------------------------------------------");
	}

	private static void result(SimulationLauncher launcher, SimulationBenchmarkResult benchmarkResult) {
		System.out.print("  ");
		String name = launcher.getClass().getSimpleName();
		System.out.print(name);
		int length1 = name.length();
		StringBuilder tabs1 = new StringBuilder();
		for (int i = 0; i < 35 - length1; i++) {
			tabs1.append(" ");
		}
		int length2 = new Integer((int) (benchmarkResult.getTime() / 1000000)).toString().length();
		StringBuilder tabs2 = new StringBuilder();
		for (int i = 0; i < 10 - length2; i++) {
			tabs2.append(" ");
		}
		System.out.print(tabs1);
		String time = benchmarkResult.getTime() > 0 ? String.valueOf(benchmarkResult.getTime() / 1000000) : "N/A";
		System.out.print(time);
		System.out.print(tabs2);
		String length = benchmarkResult.getAvgSimulationLength() > 0 ? "" + String.valueOf(benchmarkResult.getAvgSimulationLength()) : "N/A";
		System.out.print(length);
		System.out.println();
	}
}
