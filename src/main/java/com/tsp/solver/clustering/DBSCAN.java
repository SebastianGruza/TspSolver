package com.tsp.solver.clustering;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class DBSCAN {

    private double[][] dist;
    private double eps;
    private int minPts;
    private boolean[] visited;
    private List<Set<Integer>> clusters = new ArrayList<>();

    public DBSCAN(double[][] dist, double eps, int minPts) {
        this.dist = dist;
        this.eps = eps;
        this.minPts = minPts;
        this.visited = new boolean[dist.length];
    }

    public List<Set<Integer>> apply() {
        for (int i = 0; i < dist.length; i++) {
            if (!visited[i]) {
                visited[i] = true;
                List<Integer> neighbours = getNeighbours(i);
                if (neighbours.size() >= minPts) {
                    Set<Integer> cluster = new HashSet<>();
                    expandCluster(i, neighbours, cluster);
                    clusters.add(cluster);
                }
            }
        }
        return clusters;
    }

    private List<Integer> getNeighbours(int point) {
        List<Integer> neighbours = new ArrayList<>();
        for (int i = 0; i < dist.length; i++) {
            if (dist[point][i] <= eps) {
                neighbours.add(i);
            }
        }
        return neighbours;
    }

    private void expandCluster(int point, List<Integer> neighbours, Set<Integer> cluster) {
        cluster.add(point);
        int index = 0;
        while (index < neighbours.size()) {
            int neighbourPoint = neighbours.get(index);
            if (!visited[neighbourPoint]) {
                visited[neighbourPoint] = true;
                List<Integer> neighbourPoints = getNeighbours(neighbourPoint);
                if (neighbourPoints.size() >= minPts) {
                    neighbours.addAll(neighbourPoints);
                }
            }
            if (!isContainedInAnyCluster(neighbourPoint)) {
                cluster.add(neighbourPoint);
            }
            index++;
        }
    }

    private boolean isContainedInAnyCluster(int point) {
        for (Set<Integer> cluster : clusters) {
            if (cluster.contains(point)) {
                return true;
            }
        }
        return false;
    }
}
