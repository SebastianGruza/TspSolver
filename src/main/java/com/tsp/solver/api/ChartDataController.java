package com.tsp.solver.api;

import com.tsp.solver.data.Distances;
import com.tsp.solver.data.DistancesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.*;

@RestController
public class ChartDataController {

    @Autowired
    private final DistancesService distancesService;

    @Autowired
    public ChartDataController(DistancesService distancesService) {
        this.distancesService = distancesService;
    }

    @GetMapping("/chart-data")
    public Map<String, Object> getChartData() {
        Distances dist = distancesService.getCurrentDistances();
        List<Map<String, Object>> minDistances = new ArrayList<>();
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
            List<Set<Integer>> clusters;
            if (dist.clusters != null) {
                clusters = dist.clusters;
            } else {
                clusters = new ArrayList<>();
                Set<Integer> cluster = new HashSet<>();
                for (int i = 0; i < x.length; i++) {
                    cluster.add(i);
                }
                clusters.add(cluster);
            }
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
