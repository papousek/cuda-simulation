package org.sybila.ode.cpu;

import org.sybila.ode.Simulation;

public class SimulationResult implements org.sybila.ode.SimulationResult
{

	private Simulation[] simualtions;

	public SimulationResult(Simulation[] simulations) {
		if (simulations == null) {
			throw new NullPointerException("The parameter [simulations] is NULL.");
		}
		if (simulations.length == 0) {
			throw new IllegalArgumentException("The number of simulations has to be greater than zero.");
		}
		this.simualtions = simulations;
	}

	public int getNumberOfSimulations() {
		return simualtions.length;
	}

	public Simulation getSimulation(int simulationId) {
		if (simulationId < 0 || simulationId >= getNumberOfSimulations()) {
			throw new IllegalArgumentException("The simulation ID is out of the range [0," + getNumberOfSimulations() + "].");
		}
		return simualtions[simulationId];
	}

	

}
