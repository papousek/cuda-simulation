package org.sybila.ode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;
import org.apache.commons.lang.ArrayUtils;

public class EquationSystem
{

	private Collection<Equation> equations;

	private MultiAffineFunction function;

	public EquationSystem(Collection<Equation> equations) {
		if (equations == null) {
			throw new NullPointerException("The parameter [equations] is null.");
		}
		if (equations.size() <= 0) {
			throw new NullPointerException("There is no equation in equation system.");
		}
		this.equations = equations;
	}

	public MultiAffineFunction getFunction() {
		if (function == null) {
			function = createFunction();
		}
		return function;
	}

	private MultiAffineFunction createFunction() {
		SortedSet<Equation> sortedEq = new TreeSet<Equation>(equations);
		int minIndex = sortedEq.first().getLeftSide().getVariable().getIndex();
		int[] coefficientIndexes = new int[sortedEq.size() + 1];
		List<Float> coefficients = new ArrayList<Float>();
		List<Integer> factorIndexes = new ArrayList<Integer>();
		List<Integer> factors = new ArrayList<Integer>();
		int index = 0;
		coefficientIndexes[0] = 0;
		factorIndexes.add(0);
		for(Equation eq : sortedEq) {
			int numberOfAddends = 1;
			if (eq.getRightSide() instanceof Addition) {
				Expression[] sequence =  ((Addition) eq.getRightSide()).getSequence();
				numberOfAddends = sequence.length;
				for(Expression exp : sequence) {
					if (exp instanceof Coefficient) {
						coefficients.add(((Coefficient) exp).getValue());
						factorIndexes.add(factorIndexes.get(factorIndexes.size() - 1));
					}
					else {
						Expression[] mulSequence = ((Multiplication) exp).getSequence();
						coefficients.add(((Coefficient)mulSequence[0]).getValue());
						factorIndexes.add(factorIndexes.get(factorIndexes.size() - 1) + mulSequence.length - 1);
						for(int i=1; i<mulSequence.length; i++) {
							factors.add(((Variable) mulSequence[i]).getIndex() - minIndex);
						}
					}
				}
			}
			else {
				if (eq.getRightSide() instanceof Coefficient) {
					coefficients.add(((Coefficient) eq.getRightSide()).getValue());
					factorIndexes.add(factorIndexes.get(factorIndexes.size() - 1));
				}
				else {
						Expression[] mulSequence = ((Multiplication) eq.getRightSide()).getSequence();
						coefficients.add(((Coefficient)mulSequence[0]).getValue());
						factorIndexes.add(factorIndexes.get(factorIndexes.size() - 1) + mulSequence.length - 1);
						for(int i=1; i<mulSequence.length; i++) {
							factors.add(((Variable) mulSequence[i]).getIndex() - minIndex);
						}
				}
			}
			coefficientIndexes[index+1] = coefficientIndexes[index] + numberOfAddends;
			index++;
		}
		Integer[] aFI = new Integer[factorIndexes.size()];
		factorIndexes.toArray(aFI);
		Integer[] aF = new Integer[factors.size()];
		factors.toArray(aF);
		Float[] aC = new Float[coefficients.size()];
		coefficients.toArray(aC);

		return new MultiAffineFunction(ArrayUtils.toPrimitive(aC), coefficientIndexes, ArrayUtils.toPrimitive(aF), ArrayUtils.toPrimitive(aFI));
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		for(Equation eq : equations) {
			builder.append(eq.toString());
			builder.append(System.getProperty("line.separator"));
		}
		return builder.toString();
	}
}
