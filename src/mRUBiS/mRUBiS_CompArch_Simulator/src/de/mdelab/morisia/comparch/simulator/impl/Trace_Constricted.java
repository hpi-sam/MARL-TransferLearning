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

public class Trace_Constricted implements InjectionStrategy {

	private IssueType[] issueTypes;
	private Architecture eArchitecture;
	private double mean;
	private double variance;
	private Random random;
	private boolean constrict;
	private List<String> componentsGroup1;
	private List<String> componentsGroup2;
	private List<Integer> shopIDsGroup1;
	private List<Integer> shopIDsGroup2;

	public Trace_Constricted(IssueType[] issueTypes, Architecture eArchitecture, double mean, double variance,
		List<String> cg1, List<String> cg2, List<Integer> sg1, List<Integer> sg2) {
		this.issueTypes = issueTypes;
		this.eArchitecture = eArchitecture;
		this.mean = mean;
		this.variance = variance;
		this.componentsGroup1 = cg1;
        this.componentsGroup2 = cg2;
		this.shopIDsGroup1 = sg1;
		this.shopIDsGroup2 = sg2;
	}

	private List<Injection<? extends ArchitecturalElement>> createInjections(List<Integer> selectedShopIDs, List<String> componentsToUse){
        List<Injection<? extends ArchitecturalElement>> injections = new LinkedList<Injection<? extends ArchitecturalElement>>();
        for (Integer shopID : selectedShopIDs) {
			Tenant tenant = this.eArchitecture.getTenants().get(shopID);
			Component component = null;
			while (component == null) {
				int componentNumber = this.random.nextInt(tenant.getComponents().size());
				component = tenant.getComponents().get(componentNumber);
				if (component.getType().getName().equals("Future Sales Item Filter") || !componentsToUse.contains(component.getType().getName()) ){
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
		List<Integer> selectedShops1 = this.shopIDsGroup1.subList(0, numShopsGroup1);
		List<Integer> selectedShops2 = this.shopIDsGroup2.subList(0, numberOfIssues - numShopsGroup1);

        injections.addAll(this.createInjections(selectedShops1, componentsGroup1));
        injections.addAll(this.createInjections(selectedShops2, componentsGroup2));
		
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
