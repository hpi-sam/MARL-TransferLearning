package de.mdelab.morisia.comparch.simulator.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
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

public class Trace_AlternatingComponent implements InjectionStrategy {

	private IssueType[] issueTypes;
	private Architecture eArchitecture;
	private Random random = new Random(200);
    private int totalEpisodes;
    private List<List<Component>> injectionTargets;

    public void setArchitecture(Architecture architecture) {
		this.eArchitecture = architecture;
	}
    
	public Trace_AlternatingComponent(Architecture eArchitecture, int totalEpisodes) {
		this.eArchitecture = eArchitecture;
		this.totalEpisodes = totalEpisodes;
        this.initInjectionTargets();
	}

    private void initInjectionTargets() {
        this.injectionTargets = Arrays.asList(new ArrayList<Component>(), new ArrayList<Component>());
        int numberOfShops = this.eArchitecture.getTenants().size();
        List<Integer> shopIDs = IntStream.range(0, numberOfShops).boxed().collect(Collectors.toList());
		Collections.shuffle(shopIDs, this.random);

        Tenant dummyTenant = this.eArchitecture.getTenants().get(shopIDs.get(0));
        int c1Num = this.random.nextInt(dummyTenant.getComponents().size());
        int c2Num = this.random.nextInt(dummyTenant.getComponents().size());
        while (c2Num == c1Num) {
            c2Num = this.random.nextInt(dummyTenant.getComponents().size());
        }
        String c1TypeName = dummyTenant.getComponents().get(c1Num).getType().getName();
        String c2TypeName = dummyTenant.getComponents().get(c2Num).getType().getName();

        int middle = (int) (Math.floor(numberOfShops/2));

        for (int i = 0; i < middle; i++) {
			Integer shopID = shopIDs.get(i);
			Tenant tenant = this.eArchitecture.getTenants().get(shopID);
			Component c1 = tenant.getComponents().stream().filter(c -> c.getType().getName().equals(c1TypeName)).collect(Collectors.toList()).get(0);
            Component c2 = tenant.getComponents().stream().filter(c -> c.getType().getName().equals(c2TypeName)).collect(Collectors.toList()).get(0);
            this.injectionTargets.get(0).add(c1);
            this.injectionTargets.get(1).add(c2);
		}

        for (int i = middle; i < shopIDs.size(); i++) {
			Integer shopID = shopIDs.get(i);
			Tenant tenant = this.eArchitecture.getTenants().get(shopID);
			Component c1 = tenant.getComponents().stream().filter(c -> c.getType().getName().equals(c1TypeName)).collect(Collectors.toList()).get(0);
            Component c2 = tenant.getComponents().stream().filter(c -> c.getType().getName().equals(c2TypeName)).collect(Collectors.toList()).get(0);
            this.injectionTargets.get(1).add(c2);
            this.injectionTargets.get(0).add(c2);
		}
        System.out.println("Group 1");
        for(Component comp : this.injectionTargets.get(0)) {
            System.out.println(comp.getType().getName());
        }
        System.out.println("Group 2");
        for(Component comp : this.injectionTargets.get(1)) {
            System.out.println(comp.getType().getName());
        }

    }
	
	@Override
	public List<Injection<? extends ArchitecturalElement>> getInjections(int runCount) {
		List<Injection<? extends ArchitecturalElement>> injections = new LinkedList<Injection<? extends ArchitecturalElement>>();

        int idx = 0;
        if(Math.floor(totalEpisodes/2) <= runCount) {
            System.out.println("Switched components to inject into for half of the components");
            idx = 1;
        }
        List<Component> componentList = this.injectionTargets.get(idx);
        for(Component c : componentList) {
            injections.add(new Injection<Component>(IssueType.CF3, c));
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

