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
	private boolean constrict;
	private List<Components> componentsGroup1;
	private List<Components> componentsGroup2;
	private List<Integer> sg1;
	private List<Integer> sg2;

	public Trace_VariableShops(IssueType[] issueTypes, Architecture eArchitecture, double mean, double variance, boolean constrict,
		List<String> cg1, List<String> cg2, List<Integer> sg1, List<Integer> sg2) {
		this.issueTypes = issueTypes;
		this.eArchitecture = eArchitecture;
		this.mean = mean;
		this.variance = variance;
		this.constrict = constrict;
		this.initComponentGroups(cg1, cg2);
		this.sg1 = sg1;
		this.sg2 = sg2;
	}

	private  void initComponentGroups(List<String> cg1, List<String> cg2){
		// Make to actual compoentn form names
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
		Integer numShopsGroup1 = Math.floor(numberOfIssues / 2);
		List<Integer> shopsIDsGroup1 = this.sg1.subList(0, numShopsGroup1);
		List<Integer> shopsIDsGroup2 = this.sg1.subList(0, numberOfIssues - numShopsGroup1);

		for (Integer shopID : shopsIDsGroup1) {
			Tenant tenant = this.eArchitecture.getTenants().get(shopID);
			Component component = null;
			while (component == null) {
				int componentNumber = this.random.nextInt(this.componentsGroup1.size());
				component = this.componentsGroup1.get(componentNumber);
				if (component.getType().getName()
						.equals("Future Sales Item Filter")) {
					// last filter of the pipe should bot be affected by an
					// issue.
					component = null;
				}
			}
			injections.add(new Injection<Component>(IssueType.CF3, component));
		}		

		for (Integer shopID : shopsIDsGroup2) {
			Tenant tenant = this.eArchitecture.getTenants().get(shopID);
			Component component = null;
			while (component == null) {
				int componentNumber = this.random.nextInt(this.componentsGroup2.size());
				component = this.componentsGroup2.get(componentNumber);
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
