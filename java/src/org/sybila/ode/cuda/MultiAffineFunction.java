package org.sybila.ode.cuda;

import org.sybila.ode.Point;

public class MultiAffineFunction
{

	private float[] coefficients;

	private int[] coefficientIndexes;

	private int[] factors;

	private int[] factorIndexes;

	public MultiAffineFunction(float[] coefficients, int[] coefficientIndexes, int[] factors, int[] factorIndexes) {
		if (coefficients == null) {
			throw new NullPointerException("The parameter [coefficients] is NULL.");
		}
		if (coefficientIndexes == null) {
			throw new NullPointerException("The parameter [coefficientIndexes] is NULL.");
		}
		if (factors == null) {
			throw new NullPointerException("The parameter [factors] is NULL.");
		}
		if (factorIndexes == null) {
			throw new NullPointerException("The parameter [factorIndexes] is NULL.");
		}
		this.coefficients = coefficients;
		this.coefficientIndexes = coefficientIndexes;
		this.factors = factors;
		this.factorIndexes = factorIndexes;
	}
	
	public int[] getCoefficientIndexes() {
		return coefficientIndexes;
	}

	public float[] getCoefficients() {
		return coefficients;
	}

	public int[] getFactorIndexes() {
		return factorIndexes;
	}

	public int[] getFactors() {
		return factors;
	}
	

}
