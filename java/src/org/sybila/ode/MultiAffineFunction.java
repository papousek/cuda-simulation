package org.sybila.ode;

import java.util.Arrays;


/**
 * This class represents an ODE system which can be processed by the numerical
 * simulation.
 *
 * @author Jan Papousek
 */
public class MultiAffineFunction {

	private float[] coefficients;
	private int[] coefficientIndexes;
	private int[] factors;
	private int[] factorIndexes;

	/**
	 * It creates a new multiaffine function. The encoding format is described in
	 * bachelor thesis (325494 at muni.cz)
	 * @throws NullPointerException if some of the parameters is null.
	 */
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

	/**
	 * It computes X'_{dimension} a the given point X = (X_0, ..., X_n)
	 *
	 * @throws NullPointerException if the paramater point is null.
	 * @throws IllegalArgumentException if the dimension is out of the range.
	 */
	public float compute(int dimension, Point point) {
		if (point == null) {
			throw new NullPointerException("The parameter [point] is NULL.");
		}
		if (dimension < 0 && dimension >= point.getDimension()) {
			throw new IllegalArgumentException("The parameter dimension is out of the range.");
		}
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

	/**
	 * It computes X'_{dimension} a the given point X = (X_0, ..., X_n)
	 *
	 * @throws NullPointerException if the paramater point is null.
	 * @throws IllegalArgumentException if the dimension is out of the range.
	 */
	public float compute(int dimension, float[] point) {
		if (point == null) {
			throw new NullPointerException("The parameter [point] is NULL.");
		}
		if (dimension < 0 && dimension >= point.length) {
			throw new IllegalArgumentException("The parameter dimension is out of the range.");
		}
		float result = 0;
		for (int c = coefficientIndexes[dimension]; c < coefficientIndexes[dimension + 1]; c++) {
			float auxResult = coefficients[c];
			for (int f = factorIndexes[c]; f < factorIndexes[c + 1]; f++) {
				auxResult *= point[factors[f]];
			}
			result += auxResult;
		}
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

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("COEFFICIENTS:         " + Arrays.toString(coefficients) + System.getProperty("line.separator"));
		builder.append("COEFFICIENTS INDEXES: " + Arrays.toString(coefficientIndexes) + System.getProperty("line.separator"));
		builder.append("FACTORS:              " + Arrays.toString(factors) + System.getProperty("line.separator"));
		builder.append("FACTOR INDEXES:       " + Arrays.toString(factorIndexes) + System.getProperty("line.separator"));
		return builder.toString();
	}
}
