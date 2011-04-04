package org.sybila.ode;

public class Derivation implements Expression
{

	private Variable variable;

	public Derivation(Variable variable) {
		if (variable == null) {
			throw new NullPointerException("The parameter [variable] is null.");
		}
		this.variable = variable;
	}

	public Variable getVariable() {
		return variable;
	}

	public static Derivation parse(String input) {
		if (input == null) {
			throw new NullPointerException("The parameter [input] is null.");
		}
		String parsedInput = input.replace('(', ' ').replace(')', ' ').trim();
		if (!parsedInput.startsWith("d")) {
			return null;
		}
		Variable var = Variable.parse(parsedInput.substring(1));
		if (var == null) {
			return null;
		}
		return new Derivation(var);
	}

	@Override
	public String toString() {
		return "d" + variable.toString();
	}

}
