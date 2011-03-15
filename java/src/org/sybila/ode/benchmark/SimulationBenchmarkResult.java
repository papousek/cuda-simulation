package org.sybila.ode.benchmark;

public class SimulationBenchmarkResult
{

	private int avgSimulationLength;

	private long time;

	public SimulationBenchmarkResult(long time, int avgSimulationLength) {
		this.time = time;
		this.avgSimulationLength = avgSimulationLength;
	}

	public int getAvgSimulationLength() {
		return avgSimulationLength;
	}

	public long getTime() {
		return time;
	}

}
