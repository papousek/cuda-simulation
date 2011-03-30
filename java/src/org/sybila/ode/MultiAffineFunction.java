package org.sybila.ode;

public class MultiAffineFunction {

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

	public float compute(int dimension, Point point) {
		float result = 0;
		for (int c = coefficientIndexes[dimension]; c < coefficientIndexes[dimension + 1]; c++) {
			float auxResult = coefficients[c];
			for (int f = factorIndexes[c]; f < factorIndexes[c + 1]; f++) {
				auxResult *= point.getValue(factors[f]);
			}
			result += auxResult;
		}
		return result;
	}

	public float compute(int dimension, float[] data) {
//		System.out.print("X{" + dimension + "} = ");
		float result = 0;
		for (int c = coefficientIndexes[dimension]; c < coefficientIndexes[dimension + 1]; c++) {
//			if (result != 0) System.out.print(" + ");
			float auxResult = coefficients[c];
//			System.out.print("(" + auxResult + ")");
			for (int f = factorIndexes[c]; f < factorIndexes[c + 1]; f++) {
//				System.out.print(" * X{" + factors[f] + "}");
				auxResult *= data[factors[f]];
			}
			result += auxResult;
		}
//		System.out.println();
		return result;
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
