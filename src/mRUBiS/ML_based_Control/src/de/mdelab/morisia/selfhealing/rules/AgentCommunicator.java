package de.mdelab.morisia.selfhealing.rules;

import java.util.HashMap;
import java.util.concurrent.atomic.AtomicBoolean;

public class AgentCommunicator {
	
	
	public static String getMessage(AtomicBoolean reset) {
		
		String fromPython = ChunkedSocketCommunicator.readln();
		HashMap<String, String> json = ChunkedSocketCommunicator.parseJSON(new HashMap<String, String>(), fromPython);
		if (json != null && Boolean.parseBoolean(json.get("reset")) == true) {
			reset.set(true);
			ChunkedSocketCommunicator.println("resetting");
		}
		return fromPython;
	}
	
	public static HashMap<String, String> getPythonConfig() {
		String fromPython = "";
		HashMap<String, String> json = new HashMap<String, String>();
		while (true) {
			fromPython = ChunkedSocketCommunicator.readln();
			json = ChunkedSocketCommunicator.parseJSON(json, fromPython);
			if (json != null) {
				return json;
			}
		}
	}
	
	public static HashMap<String, String> waitForReset() {
		while (true) {
			String fromPython = ChunkedSocketCommunicator.readln();
			HashMap<String, String> json = ChunkedSocketCommunicator.parseJSON(new HashMap<String, String>(), fromPython);
			if (json != null && Boolean.parseBoolean(json.get("reset")) == true) {
				return json;
			}
		}
	}
}
