package org.sybila.ode;

public class Multiplication extends AbstractBinaryOperator
{

	public Multiplication(Expression left, Expression right) {
		super(left, right);
	}

	public static Multiplication parse(String input) {
		if (input == null) {
			throw new NullPointerException("The parameter [input] is null.");
		}
		String[] factors = input.replace('(', ' ').replace(')', ' ').trim().split("\\*");
		if (factors.length < 2) {
			return null;
		}
		Expression[] exps = new Expression[factors.length];
		for(int i=0; i<factors.length; i++) {
			Coefficient c = Coefficient.parse(factors[i]);
			if (c != null) {
				exps[i] = c;
			}
			else {
				Variable v = Variable.parse(factors[i]);
				if (v != null) {
					exps[i] = v;
				}
				else {
					return null;
				}
			}
		}
		Multiplication multi = new Multiplication(exps[0], exps[1]);
		for(int i=2; i<exps.length; i++) {
			multi = new Multiplication(multi, exps[i]);
		}
		return multi;
	}

	protected String getStringOperator() {
		return "*";
	}

}
