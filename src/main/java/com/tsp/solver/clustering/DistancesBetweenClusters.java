package com.tsp.solver.clustering;

import java.util.*;


public class DistancesBetweenClusters {

    class PointPairComparator implements Comparator<PointPair> {
        public int compare(PointPair a, PointPair b) {
            if (a.distance < b.distance) return -1;
            if (a.distance > b.distance) return 1;
            return 0;
        }
    }

    public List<List<PointPair>> findMinInterClusterDistances(List<Set<Integer>> clusters, double[][] dist, int k) {
        List<List<PointPair>> result = new ArrayList<>();

        for (int i = 0; i < clusters.size(); i++) {
            System.out.println("Cluster i = " + i);
            for (int j = i + 1; j < clusters.size(); j++) {
                PriorityQueue<PointPair> queue = new PriorityQueue<>(new PointPairComparator());

                for (int point1 : clusters.get(i)) {
                    for (int point2 : clusters.get(j)) {
                        queue.add(new PointPair(point1, point2, dist[point1][point2]));
                    }
                }

                List<PointPair> minPairs = new ArrayList<>();
                for (int m = 0; m < k && !queue.isEmpty(); m++) {
                    minPairs.add(queue.poll());
                }

                result.add(minPairs);
            }
        }

        return result;
    }

    public List<PointPair> findKNearestNeighbors(double[][] dist, int k) {
        List<PointPair> result = new ArrayList<>();

        for (int i = 0; i < dist.length; i++) {
            PriorityQueue<PointPair> queue = new PriorityQueue<>(new PointPairComparator());

            for (int j = 0; j < dist.length; j++) {
                if (i != j) {
                    queue.add(new PointPair(i, j, dist[i][j]));
                }
            }

            for (int m = 0; m < k && !queue.isEmpty(); m++) {
                result.add(queue.poll());
            }
        }

        return result;
    }

    public double[] kthNearestNeighborDistances(double[][] dist, int k) {
        int n = dist.length;
        double[] kDistances = new double[n];

        for (int i = 0; i < n; i++) {
            PriorityQueue<Double> queue = new PriorityQueue<>();

            for (int j = 0; j < n; j++) {
                if (i != j) {
                    queue.add(dist[i][j]);
                }
            }

            for (int m = 0; m < k; m++) {
                if (!queue.isEmpty()) {
                    kDistances[i] = queue.poll();
                }
            }
        }
        return kDistances;
    }


}
