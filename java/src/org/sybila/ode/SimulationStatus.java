package org.sybila.ode;

public enum SimulationStatus
{

	OK, TIMEOUT, PRECISION;

	public static SimulationStatus fromInt(int status) {
		switch(status) {
			case 0:
				return OK;
			case 1:
				return TIMEOUT;
			case 2:
				return PRECISION;
			default:
				//return null;
				throw new IllegalStateException("There is no status corresponding to the number [" + status + "].");
		}
	}

}
