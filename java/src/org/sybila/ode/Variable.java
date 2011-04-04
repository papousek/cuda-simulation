package org.sybila.ode;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

public class Variable implements Expression
{

	private int index;

	private static Map<String, Integer> indexes = new HashMap<String, Integer>();;

	private static int lastIndex = -1;

	private String name;

	public Variable(int index, String name) {
		if (index < 0) {
			throw new IllegalArgumentException("The index of variable has to be a non negative number.");
		}
		if (name == null) {
			throw new NullPointerException("The parameter [name] is null.");
		}
		this.index = index;
		this.name = name;
	}

	public int getIndex() {
		return index;
	}

	public String getName() {
		return name;
	}

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
