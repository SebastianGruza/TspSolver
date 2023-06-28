package com.tsp.solver.clustering;

import com.tsp.solver.data.Distances;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class ClusteringAndKnn {

    @Autowired
    private Distances dist;

    private List<PointPair> getPairs() {
        int k = 10;
        DistancesBetweenClusters distancesBetweenClusters = new DistancesBetweenClusters();
        List<List<PointPair>> pointPairs = distancesBetweenClusters.findMinInterClusterDistances(dist.clusters, dist.distances, k);
        List<PointPair> result = distancesBetweenClusters.findKNearestNeighbors(dist.distances, k);
        result.addAll(pointPairs.stream().flatMap(List::stream).collect(Collectors.toList()));
        return result;
    }
}
