package org.sybila.ode;

public enum SimulationStatus
{

	OK, TIMEOUT;

	public static SimulationStatus fromInt(int status) {
		switch(status) {
			case 0:
				return OK;
			case 1:
				return TIMEOUT;
			default:
				throw new IllegalStateException("There is no status corresponding to the number [" + status + "].");
		}
	}

}
