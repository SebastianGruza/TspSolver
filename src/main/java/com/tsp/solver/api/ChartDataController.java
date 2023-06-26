package com.tsp.solver.api;

import com.tsp.solver.clustering.DistancesBetweenClusters;
import com.tsp.solver.data.Distances;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.*;

@RestController
public class ChartDataController {

    @Autowired
    private final Distances dist;

    public ChartDataController(Distances dist) {
        this.dist = dist;
    }


    @GetMapping("/chart-data")
    public Map<String, Object> getChartData() {

        Integer counterTotal = 0;

        //Distances dist = new Distances("gr9882.tsp", 7);
        //Distances dist = new Distances("rl5934.tsp", 7);
        //Distances dist = new Distances("PL4000.txd");
        int minPts = 1;
        DistancesBetweenClusters distancesBetweenClusters = new DistancesBetweenClusters();
//        for (int k = 1; k < 10; k++) {
//            double[] kDistances = distancesBetweenClusters.kthNearestNeighborDistances(dist.distances, k);
//            Arrays.sort(kDistances);
//            Double meanKdistance = Arrays.stream(kDistances).average().getAsDouble();
//            Double stdDevKdistance = Math.sqrt(Arrays.stream(kDistances).map(d -> Math.pow(d - meanKdistance, 2)).average().getAsDouble());
//            System.out.println("K: " + k);
//            System.out.println("Mean k-distance: " + meanKdistance);
//            System.out.println("Std dev k-distance: " + stdDevKdistance);
//            System.out.println("Min k-distance: " + kDistances[0]);
//            System.out.println("Max k-distance: " + kDistances[kDistances.length - 1]);
//        }

//        double[] kDistances = distancesBetweenClusters.kthNearestNeighborDistances(dist.distances, 6);
//        Arrays.sort(kDistances);
//        Double meanKdistance = Arrays.stream(kDistances).average().getAsDouble();
//        Double eps = meanKdistance * 1.5;
//        DBSCAN dbscan = new DBSCAN(dist.distances, eps, minPts);
//        List<Set<Integer>> clusters = dbscan.apply();
//        System.out.println("Groups: " + clusters.size());
//        System.out.println("Total points in clusters: " + clusters.stream().mapToInt(Set::size).sum());
//        double[] x = dist.x;
//        double[] y = dist.y;

//        List<List<DistancesBetweenClusters.PointPair>> pointPairs = distancesBetweenClusters.findMinInterClusterDistances(clusters, dist.distances, 1);
//        List<DistancesBetweenClusters.PointPair> kNearestNeighbors = distancesBetweenClusters.findKNearestNeighbors(dist.distances, 1);
        List<Map<String, Object>> minDistances = new ArrayList<>();
//
//        for (List<DistancesBetweenClusters.PointPair> pairs : pointPairs) {
//            for (DistancesBetweenClusters.PointPair pair : pairs) {
//                Map<String, Object> minPair = new HashMap<>();
//                minPair.put("point1", pair.point1);
//                minPair.put("point2", pair.point2);
//                minPair.put("distance", pair.distance);
//                minDistances.add(minPair);
//            }
//        }
//
//        for (DistancesBetweenClusters.PointPair pair : kNearestNeighbors) {
//            Map<String, Object> minPair = new HashMap<>();
//            minPair.put("point1", pair.point1);
//            minPair.put("point2", pair.point2);
//            minPair.put("distance", pair.distance);
//            minDistances.add(minPair);
//        }

        List<Set<Integer>> clusters = new ArrayList<>();
        if (dist != null) {
            double[] x = dist.x;
            double[] y = dist.y;
            if (dist.bestSolution != null) {
                //explode bestSolution by - into pairs
                String[] pairs = dist.bestSolution.split("-");
                //create a minDistances list by pairs
                for (int i = 0; i < pairs.length - 1; i++) {
                    Map<String, Object> minPair = new HashMap<>();
                    minPair.put("point1", Integer.parseInt(pairs[i]));
                    minPair.put("point2", Integer.parseInt(pairs[i + 1]));
                    minPair.put("distance", dist.distances[Integer.parseInt(pairs[i])][Integer.parseInt(pairs[i + 1])]);
                    minDistances.add(minPair);
                }
                //add first and last point
                Map<String, Object> minPair = new HashMap<>();
                minPair.put("point1", Integer.parseInt(pairs[0]));
                minPair.put("point2", Integer.parseInt(pairs[pairs.length - 1]));
                minPair.put("distance", dist.distances[Integer.parseInt(pairs[0])][Integer.parseInt(pairs[pairs.length - 1])]);
                minDistances.add(minPair);
            }

            Map<String, Object> data = new HashMap<>();
            Set<Integer> cluster = new HashSet<>();
            for (int i = 0; i < x.length; i++) {
                cluster.add(i);
            }
            clusters.add(cluster);
            data.put("clusters", clusters);
            data.put("x", x);
            data.put("y", y);
            data.put("minDistances", minDistances);
            return data;
        } else {
            return null;
        }

    }
}
