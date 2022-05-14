package de.mdelab.morisia.selfhealing.rules;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.HashMap;
import java.util.logging.Logger;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import mRUBiS.Observations.Observations;

public class ChunkedSocketCommunicator {
	
	// Open up the socket
	private static int port = 8080;
	
	private static ServerSocket server;
	private static Socket client;
	private static BufferedReader in;
    private static PrintWriter out;
    private static PrintWriter logger;
    
    private final static Integer MAX_LENGTH = 8192;
    
    private static void ensureSocketIsOpen() {
    	if (server == null || (server != null && !server.isBound())) {
        	try {
    			server = new ServerSocket(port);
    			System.out.println("wait for connection on port " + port);
    			client = server.accept();
    			System.out.println("got connection on port " + port);
    			in = new BufferedReader(new InputStreamReader(client.getInputStream()));
    			out = new PrintWriter(client.getOutputStream(),true);
    			logger = new PrintWriter("log.txt", "UTF-8");
    		} catch (IOException e) {
    			System.out.println("Problem with opening the socket");
    			e.printStackTrace();
    		}	
    	}
    }
    
    private final static Logger LOGGER = Logger.getLogger(RuleSelector.class
			.getName());
    
    public static String readln() {
    	ensureSocketIsOpen();
    	
    	try {
			String message = in.readLine();
			if (message.equals("exit")) {
				System.out.println("Received 'exit' signal from Python, exiting...");
				close();
				System.exit(1);
			}
			return message;
		} catch (IOException e) {
			e.printStackTrace();
			close();
			return null;
		}
    }
    
    public static void println(String message) {
    	ensureSocketIsOpen();
    	
    	if (message.length() >= MAX_LENGTH) {
    		printChunked(message);
    	}
    	else {        	
        	out.println(message);
    	}
    	logger.println(message);
    }
    
    private static void printChunked(String message) {
    	Integer numChunks = (int) Math.ceil(message.length() / (double)MAX_LENGTH);
    	HashMap<String, Integer> chunkMessage = new HashMap<String, Integer>();
    	chunkMessage.put("number_of_chunks", numChunks);
    	
    	String json = "";
		try {
			json = new ObjectMapper().writeValueAsString(chunkMessage);
		} catch (JsonProcessingException e) {
			e.printStackTrace();
		}
		
		out.println(json);
		
		waitForMessage("send_chunks");
		
		for (int i = 0; i < numChunks; i++) {
			String partMessage = message.substring(i * MAX_LENGTH, Math.min((i + 1) * MAX_LENGTH, message.length()));
			out.println(partMessage);
			waitForMessage("received");
		}
		
		out.println("finished_sending");
    }
    
    public static void log(String message) {
    	logger.println(message);
    }
    
    public static void close() {
    	logger.println("closed");
    	out.close();
    	logger.close();
    	try {
			server.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
    }
    
    public static <T> HashMap<String, T> readJSON(HashMap<String, T> json) {
    	ensureSocketIsOpen();
    	String fromPython = "";
		while(true) {
			fromPython = readln();
			json = parseJSON(json, fromPython);
			return json;
		}
    }
    
    public static void waitForMessage(String message) {
    	String fromPython = "";
		// System.out.println("Waiting for Python to send '" + message + "'...");
		while(true) { 
			fromPython = readln();
			if (fromPython.equals(message)) {
				return;
			}
		}
    }
    
    public static <T> HashMap<String, T> parseJSON(HashMap<String, T> json, String message) {
    	try {
			json = new ObjectMapper().readValue(message, HashMap.class);
			logger.println(json);	
			return json;			
		} catch (IOException e) {
			System.out.println("Did not receive valid json from Python:");
			System.out.println(message);
		}
		return null;
    }
}
