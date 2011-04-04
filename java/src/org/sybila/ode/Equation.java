package org.sybila.ode;

public class Equation implements Expression, Comparable<Equation>
{

	private Derivation left;

	private Expression right;

	public Equation(Derivation left, Expression right) {
		if (left == null) {
			throw new NullPointerException("The parameter [left] is null.");
		}
		if (right == null) {
			throw new NullPointerException("The parameter [right] is null.");
		}
		this.left = left;
		this.right = right;
	}

	public Derivation getLeftSide() {
		return left;
	}

	public Expression getRightSide() {
		return right;
	}

	public static Equation parse(String input) {
		if (input == null) {
			throw new NullPointerException("The parameter [input] is null.");
		}
		String[] sides = input.replace('(', ' ').replace(')', ' ').trim().split("=");
		if (sides.length < 2) {
			return null;
		}
		Derivation der	= Derivation.parse(sides[0]);
		Coefficient c	= Coefficient.parse(sides[1]);
		if (c != null) {
			return new Equation(der, c);
		}
		Variable v		= Variable.parse(sides[1]);
		if (v != null) {
			return new Equation(der, v);
		}
		Multiplication mul = Multiplication.parse(sides[1]);
		if (mul != null) {
			return new Equation(der, mul);
		}
		Addition add	= Addition.parse(sides[1]);
		if (add != null) {
			return new Equation(der, add);
		}
		return null;
	}

	@Override
	public String toString() {
		return left + " = " + right;
	}

	public int compareTo(Equation e) {
		return getLeftSide().getVariable().getIndex() - e.getLeftSide().getVariable().getIndex();
	}

}
