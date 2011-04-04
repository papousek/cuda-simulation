package org.sybila.ode;

public class Coefficient implements Expression
{

	private float value;

	public Coefficient(float value) {
		this.value = value;
	}

	public float getValue() {
		return value;
	}

	public static Coefficient parse(String input) {
		if (input == null) {
			throw new NullPointerException("The parameter [input] is null.");
		}
		try {
			return new Coefficient(Float.valueOf(input.replace('(', ' ').replace(')', ' ').trim()));
		}
		catch (NumberFormatException e) {
			return null;
		}
	}

	@Override
	public String toString() {
		return String.valueOf(value);
	}

}
