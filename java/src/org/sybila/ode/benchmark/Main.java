package org.sybila.ode.benchmark;

import org.sybila.ode.SimulationLauncher;
import org.sybila.ode.cpu.CpuEulerSimulationLauncher;
import org.sybila.ode.cpu.CpuRkf45SimulationLauncher;
import org.sybila.ode.cuda.CudaSimulationLauncher;

public class Main
{

	private static final int MIN_DIMENSION = 10;

	private static final int DIMENSION_STEP = 10;

	private static final int MAX_DIMENSION = 20;

	private static final int MAX_NUMBER_OF_SIMULATIONS = 500;

	private static final int MIN_NUMBER_OF_SIMULATIONS = 500;

	private static final int NUMBER_OF_SIMULATIONS_STEP = 100;

	public static void main(String[] args) {
		header();
		SimulationLauncher[] launchers = new SimulationLauncher[3];
		for(int numberOfSimulations=MIN_NUMBER_OF_SIMULATIONS; numberOfSimulations<=MAX_NUMBER_OF_SIMULATIONS; numberOfSimulations+=NUMBER_OF_SIMULATIONS_STEP)	{
			for(int dim=MIN_DIMENSION; dim<=MAX_DIMENSION; dim+=DIMENSION_STEP) {
				int maxSize = 100000000 / (numberOfSimulations * dim);
				SimulationBenchmark benchmark = new SimulationBenchmark(dim, numberOfSimulations, maxSize);
				launchers[0] = new CpuEulerSimulationLauncher(dim, numberOfSimulations, maxSize, benchmark.getFunction());
				launchers[1] = new CpuRkf45SimulationLauncher(dim, numberOfSimulations, maxSize, benchmark.getFunction());
				launchers[2] = new CudaSimulationLauncher(dim, numberOfSimulations, maxSize, benchmark.getFunction());
				line();
				System.out.println("  DIMENSION:             " + dim);
				System.out.println("  NUMBER OF SIMULATIONS: " + numberOfSimulations);
				System.out.println("  MAX SIZE:              " + maxSize);
				line();
				for(SimulationLauncher launcher : launchers) {
					result(launcher, benchmark.time(launcher) / 1000000);
					System.gc();
				}
				((CudaSimulationLauncher) launchers[2]).destroy();
				for(int i=0; i<launchers.length; i++) {
					launchers[i] = null;
				}
				System.gc();
			}
		}
	}

	private static void header() {
		line();
		System.out.println("  MINIMUM DIMENSION: " + MIN_DIMENSION);
		System.out.println("  MAXIMUM DIMENSION: " + MAX_DIMENSION);
		System.out.println("  DIMENSION STEP:    " + DIMENSION_STEP);
		System.out.println("  MINIMUM NUMBER OF SIMULATIONS: " + MIN_NUMBER_OF_SIMULATIONS);
		System.out.println("  MAXIMUM NUMBER OF SIMULATIONS: " + MAX_NUMBER_OF_SIMULATIONS);
		System.out.println("  NUMBER OF SIMULATIONS STEP:    " + NUMBER_OF_SIMULATIONS_STEP);
		line();
	}

	private static void line() {
		System.out.println("--------------------------------------------------------------------------------");
	}

	private static void result(SimulationLauncher launcher, double time) {
		System.out.print("  ");
		String name = launcher.getClass().getSimpleName();
		System.out.print(name);
		int length1	= name.length();
		StringBuilder tabs1 = new StringBuilder();
		for (int i=0; i<35-length1;i++) {
			tabs1.append(" ");
		}
		System.out.print(tabs1);
		System.out.print(time);
		System.out.println();
	}
}