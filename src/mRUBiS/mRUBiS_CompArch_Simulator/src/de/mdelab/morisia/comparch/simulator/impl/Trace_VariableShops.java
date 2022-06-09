package de.mdelab.morisia.comparch.simulator.impl;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import de.mdelab.morisia.comparch.ArchitecturalElement;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.Injection;
import de.mdelab.morisia.comparch.simulator.InjectionStrategy;
import de.mdelab.morisia.comparch.simulator.IssueType;

public class Trace_VariableShops implements InjectionStrategy {

	private IssueType[] issueTypes;
	private Architecture eArchitecture;
	private double mean;
	private double variance;
	private Random random;

	public Trace_VariableShops(IssueType[] issueTypes, Architecture eArchitecture, double mean, double variance) {
		this.issueTypes = issueTypes;
		this.eArchitecture = eArchitecture;
		this.mean = mean;
		this.variance = variance;
	}
	
	@Override
	public List<Injection<? extends ArchitecturalElement>> getInjections(int runCount) {
		List<Injection<? extends ArchitecturalElement>> injections = new LinkedList<Injection<? extends ArchitecturalElement>>();

//		this.random = new Random(this.eArchitecture.getTenants().size() * 10 + (runCount % 10));
		this.random = new Random(runCount);
		int numberOfShops = this.eArchitecture.getTenants().size();
		int numberOfIssues = (int) Math.round(this.random.nextGaussian() * this.variance + this.mean);
		numberOfIssues = Math.min(Math.max(1, numberOfIssues), numberOfShops);
		List<Integer> shopIDs = IntStream.range(0, numberOfShops).boxed().collect(Collectors.toList());
		Collections.shuffle(shopIDs, this.random);
		for (int i = 0; i < numberOfIssues; i++) {
			Integer shopID = shopIDs.get(i);
			Tenant tenant = this.eArchitecture.getTenants().get(shopID);
			Component component = null;
			while (component == null) {
				int componentNumber = this.random.nextInt(tenant.getComponents().size());
				component = tenant.getComponents().get(componentNumber);
				if (component.getType().getName()
						.equals("Future Sales Item Filter")) {
					// last filter of the pipe should bot be affected by an
					// issue.
					component = null;
				}
			}
			injections.add(new Injection<Component>(IssueType.CF3, component));
		}		
		
		return injections;
	}

	@Override
	public void notifyAboutSuccess(
			List<Injection<? extends ArchitecturalElement>> injections) {
		for (Injection<? extends ArchitecturalElement> i : injections) {
			if (!i.isSuccess()) {
				String message = "The simulator could not successfully inject a "
						+ i.getIssueType()
						+ " issue to element "
						+ i.getTarget();
				System.err.println(message);
				throw new RuntimeException(message);
			}
		}
	}
}
