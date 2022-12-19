package com.tsp.solver;

import java.util.HashSet;
import java.util.Set;

public class GreedyAlgorithm {

    public static void CreateNewGenerationWithGreedyAlgorithm(int number, int noThread, double[][] distances, int[][] path, int ts) {
        int start = 0; //number * noThread;

        for (int noPath = start; noPath < start + number; noPath++) {
            Set<Integer> cityVisited = new HashSet<>();
            cityVisited.add(noPath);
            double best;
            int numberBest = noPath;
            path[noPath][0] = numberBest;
            for (int cityNumber = 1; cityNumber < distances.length; cityNumber++) {
                int previousBest = numberBest;
                best = Double.MAX_VALUE;
                for (int j = 0; j < distances.length; j++) {
                    if (!cityVisited.contains(j)) {
                        double actual = distances[previousBest][j];
                        if (actual < best) {
                            numberBest = j;
                            best = actual;
                        }
                    }
                }
                cityVisited.add(numberBest);
                path[noPath][cityNumber] = numberBest;
            }
            System.out.println("Number path: " + noPath);
        }
        for (int noPath = start + number; noPath < ts ; noPath++) {
            for (int numer = 0; numer < distances.length; numer++) {
                path[noPath][numer] = path[noPath % number][numer];
            }
            System.out.println("Number path: " + noPath);
        }
    }
}
