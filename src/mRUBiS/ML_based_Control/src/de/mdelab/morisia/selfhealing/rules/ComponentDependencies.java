package de.mdelab.morisia.selfhealing.rules;

import java.util.ArrayList;
import java.util.Hashtable;

public class ComponentDependencies {
	
	public ComponentDependencies() {
		
	}
	
	public static Double getFixFailProbability(String typeOfComponentToFix, ArrayList<String> failingComponentTypes) {
		Hashtable<String, Double> probabilities = TransitionMatrixHandler.getTransitionMatrix().get(typeOfComponentToFix);
		Double sum = 0.0;
		for (String failing : failingComponentTypes) {
			sum += probabilities.get(failing);
		}
		return sum;
	}
}
