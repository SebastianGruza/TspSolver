package com.tsp.solver.data;

import java.util.HashMap;
import java.util.Map;

public class Colony {
    Map<Path, int[]> individuals;
    Double best;
    Double worst;

    public Colony() {
        this.individuals = new HashMap<>();
        this.best = Double.MAX_VALUE;
        this.worst = Double.MIN_VALUE;
    }

    public Map<Path, int[]> getIndividuals() {
        return individuals;
    }

    public Double getBest() {
        return best;
    }

    public Double getWorst() {
        return worst;
    }



    public Colony(Map<Path, int[]> individuals, Double best, Double worst) {
        this.individuals = individuals;
        this.best = best;
        this.worst = worst;
    }

    public Colony(Colony colonyFirst, Colony colonySecond) {
        this.individuals = new HashMap<>();
        this.individuals.putAll(colonyFirst.getIndividuals());
        this.individuals.putAll(colonySecond.getIndividuals());
        this.best = colonyFirst.getBest() > colonySecond.getBest() ? colonySecond.getBest() : colonyFirst.getBest();
        this.worst = colonyFirst.getWorst() < colonySecond.getWorst() ? colonySecond.getWorst() : colonyFirst.getWorst();
    }
}
