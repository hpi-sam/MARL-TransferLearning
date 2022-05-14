package de.mdelab.morisia.comparch.simulator.impl;

import java.util.LinkedList;
import java.util.List;

import de.mdelab.morisia.comparch.ArchitecturalElement;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.Injection;
import de.mdelab.morisia.comparch.simulator.InjectionStrategy;
import de.mdelab.morisia.comparch.simulator.IssueType;

public class Trace_SpecificComponent implements InjectionStrategy {

	private Architecture eArchitecture;
	private String componentName;

	public Trace_SpecificComponent(Architecture eArchitecture, String componentName) {
		this.eArchitecture = eArchitecture;
		this.componentName = componentName;
	}
	
	@Override
	public List<Injection<? extends ArchitecturalElement>> getInjections(int runCount) {
		List<Injection<? extends ArchitecturalElement>> injections = new LinkedList<Injection<? extends ArchitecturalElement>>();

		
		
		for (Tenant tenant : this.eArchitecture.getTenants()) {
			Component component = this.findComponent(tenant);
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
	
	
	private Component findComponent(Tenant tenant) {
		
		for (Component component : tenant.getComponents()) {
			if (component.getType().getName().equals(this.componentName)) {
				return component;
			}
		}
		throw new RuntimeException("Component Name not valid");
	}
}
