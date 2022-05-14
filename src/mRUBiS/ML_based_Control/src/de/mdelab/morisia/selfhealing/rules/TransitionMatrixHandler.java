package de.mdelab.morisia.selfhealing.rules;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.*;
import java.util.Hashtable;

public final class TransitionMatrixHandler {
	
	private static Hashtable<String, Hashtable<String, Double>> transitionMatrix = initializeMatrix();
	
	private TransitionMatrixHandler() {
		
	}
	
	private static Hashtable<String, Hashtable<String, Double>> initializeMatrix() {
		return initializeMatrix(Paths.get(System.getProperty("user.dir")).getParent().getParent().resolve("mrubis_controller").resolve("transition_matrix.csv"));
	}
	
	private static Hashtable<String, Hashtable<String, Double>> initializeMatrix(Path transitionMatrixPath) {
		Hashtable<String, Hashtable<String, Double>> newTransitionMatrix = new Hashtable<String, Hashtable<String, Double>>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(transitionMatrixPath.toString()));
			String line = br.readLine();
			String[] keys = line.split(",");
			while ((line = br.readLine()) != null) {
				String[] values = line.split(",");
				Hashtable<String, Double> currentLine = new Hashtable<String, Double>();
				
				for (int j = 1; j < values.length; j++) {
					currentLine.put(keys[j], Double.parseDouble(values[j]));
				}				
				
				newTransitionMatrix.put(values[0], currentLine);		
			}
		} catch (FileNotFoundException e) {
			System.out.println("File " + transitionMatrixPath.toString() + " not found.");
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return newTransitionMatrix;
	}
	
	public static Hashtable<String, Hashtable<String, Double>> getTransitionMatrix() {
		return transitionMatrix;
	}
}
