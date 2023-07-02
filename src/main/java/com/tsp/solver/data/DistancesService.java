package com.tsp.solver.data;

public class DistancesService {

    private Distances currentDistances;

    public void updateDistances(Distances newDistances) {
        this.currentDistances = newDistances;
    }

    public Distances getCurrentDistances() {
        return this.currentDistances;
    }
}
