package org.sybila.ode;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * This class represents a variable in an ODE system
 *
 * @author Jan Papousek
 */
public class Variable implements Expression
{

	/**
	 * Each variable has an index which should be a unique identificator
	 * in the ODE system
	 */
	private int index;

	/**
	 * This map provides retrieving variable indexes by their name
	 */
	private static Map<String, Integer> indexes = new HashMap<String, Integer>();;

	/**
	 * Last generated index
	 */
	private static int lastIndex = -1;

	/**
	 * Name of the variable. The only purpose of this attribute is textual
	 * representation of the object
	 */
	private String name;

	/**
	 * It creates a new variable with the given index and name
	 * 
	 * @param index non-negative number which unique in the ODE system
	 * @param name textual representation of the variable
	 * @throws IllegalArgumentException if the index is negative
	 * @throws NullPointerExceptionthe if the name is null
	 */
	private Variable(int index, String name) {
		if (index < 0) {
			throw new IllegalArgumentException("The index of variable has to be a non negative number.");
		}
		if (name == null) {
			throw new NullPointerException("The parameter [name] is null.");
		}
		this.index = index;
		this.name = name;
	}

	/**
	 * It returns variable index
	 */
	public int getIndex() {
		return index;
	}

	/**
	 * It returns variable name
	 */
	public String getName() {
		return name;
	}

	/**
	 * It tries to parse the given string and return a new variable.
	 * 
	 * @param input
	 * @return a new variable or null if the given string can not be parsed
	 * @throws NullPointerException if the input is null
	 */
	public static Variable parse(String input) {
		if (input == null) {
			throw new NullPointerException("The parameter [input] is null.");
		}
		String parsedInput = input.replace('(', ' ').replace(')', ' ').trim().toUpperCase();
		if (parsedInput.split(" ").length > 1) {
			return null;
		}
		if (!Pattern.compile("^[A-Z]").matcher(parsedInput).find()) {
			return null;
		}
		if (!indexes.containsKey(parsedInput)) {
			lastIndex++;
			indexes.put(parsedInput, lastIndex);
		}
		return new Variable(indexes.get(parsedInput), parsedInput);
	}

	/**
	 * It generates a new variable
	 */
	public static Variable generate() {
		lastIndex++;
		indexes.put("X_" + lastIndex, lastIndex);
		return new Variable(lastIndex, "X_" + lastIndex);
	}

	@Override
	public String toString() {
		return name;
	}

}
