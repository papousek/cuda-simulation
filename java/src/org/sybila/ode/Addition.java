package org.sybila.ode;

public class Addition extends AbstractBinaryOperator
{

	public Addition(Expression left, Expression right) {
		super(left, right);
	}

	public static Addition parse(String input) {
		if (input == null) {
			throw new NullPointerException("The parameter [input] is null.");
		}
		String[] addends = input.replace('(', ' ').replace(')', ' ').trim().split("\\+");
		if (addends.length < 2) {
			return null;
		}
		Expression[] exps = new Expression[addends.length];
		for(int i=0; i<addends.length; i++) {
			Coefficient c = Coefficient.parse(addends[i]);
			if (c != null) {
				exps[i] = c;
			}
			else {
				Variable v = Variable.parse(addends[i]);
				if (v != null) {
					exps[i] = v;
				}
				else {
					Multiplication m = Multiplication.parse(addends[i]);
					if (m != null) {
						exps[i] = m;
					}
					else {
						return null;
					}
				}
			}
		}
		Addition add = new Addition(exps[0], exps[1]);
		for (int i=2; i<exps.length; i++) {
			add = new Addition(add, exps[i]);
		}
		return add;
	}

	protected String getStringOperator() {
		return "+";
	}

}
