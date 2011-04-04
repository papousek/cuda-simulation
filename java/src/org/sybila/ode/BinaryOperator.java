package org.sybila.ode;

public interface BinaryOperator extends Expression
{

	Expression getLeftSide();

	Expression getRightSide();

	Expression[] getSequence();
}
