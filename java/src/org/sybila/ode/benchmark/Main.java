package org.sybila.ode.benchmark;

import org.sybila.ode.SimulationLauncher;
import org.sybila.ode.cpu.CpuSimulationLauncher;
import org.sybila.ode.cuda.CudaSimulationLauncher;

public class Main
{

	private static final int DIMENSION = 10;

	private static final int MAX_SIZE = 100000;

	private static final int NUMBER_OF_SIMULATIONS = 100;

	public static void main(String[] args) {
		header();
		SimulationBenchmark benchmark = new SimulationBenchmark(DIMENSION, NUMBER_OF_SIMULATIONS, MAX_SIZE);
		SimulationLauncher[] launchers = new SimulationLauncher[] {
			new CpuSimulationLauncher(DIMENSION, NUMBER_OF_SIMULATIONS, MAX_SIZE, benchmark.getFunction()),
			new CudaSimulationLauncher(DIMENSION, NUMBER_OF_SIMULATIONS, MAX_SIZE, benchmark.getFunction())
		};
		for(SimulationLauncher launcher : launchers) {
			result(launcher, benchmark.time(launcher) / 1000000);
		}
	}

	private static void header() {
		line();
		System.out.println("  SIMULATION BENCHMARK");
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