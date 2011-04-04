package org.sybila.ode;

import java.util.ArrayList;
import java.util.List;

abstract public class AbstractBinaryOperator implements BinaryOperator
{
	private Expression left;

	private Expression right;

	public AbstractBinaryOperator(Expression left, Expression right) {
		if (left == null) {
			throw new NullPointerException("The parameter [left] is null.");
		}
		if (right == null) {
			throw new NullPointerException("The parameter [right] is null.");
		}
		this.left = left;
		this.right = right;
	}

	public Expression getLeftSide() {
		return left;
	}

	public Expression getRightSide() {
		return right;
	}

	public Expression[] getSequence() {
		List<Expression> seq = new ArrayList<Expression>();
		if (left.getClass() == this.getClass()) { // HACK
			for (Expression exp : ((BinaryOperator) left).getSequence()) {
				seq.add(exp);
			}
		}
		else {
			seq.add(left);
		}
		if (right.getClass() == this.getClass()) { // HACK
			for (Expression exp : ((BinaryOperator) right).getSequence()) {
				seq.add(exp);
			}
		}
		else {
			seq.add(right);
		}
		Expression[] result = new Expression[seq.size()];
		seq.toArray(result);
		return result;
	}

	@Override
	public String toString() {
		return getLeftSide() + " " + getStringOperator() + " " + getRightSide();
	}

	abstract protected String getStringOperator();
}
