package com.tsp.solver.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Distances {
    public final int n;
    public final double[][] distances;
    final String filename;

    public Distances(String filename) {
        this.filename = filename;
        List<String> allLines = LoadFromFile(filename);
        List<Double> xCoord = new ArrayList<>();
        List<Double> yCoord = new ArrayList<>();
        int actualLine = 0;
        for (String line : allLines) {
            if (line != null) {
                String[] split = line.split(";");
                if (split.length == 3) {
                    try {
                        xCoord.add(Double.parseDouble(split[1]));
                        yCoord.add(Double.parseDouble(split[2]));
                        actualLine += 1;
                    } catch (NumberFormatException nfe) {

                    }
                }
            }
        }
        this.n = actualLine;
        this.distances = new double[this.n][this.n];
        for (int i = 0; i < actualLine; i++) {
            for (int j = 0; j < actualLine; j++) {
                distances[i][j] = Math.sqrt(Math.pow(xCoord.get(i) - xCoord.get(j), 2) + Math.pow(yCoord.get(i) - yCoord.get(j), 2));
            }
        }
    }

//    public Distances(String filename, int startFrom) {
//        this.filename = filename;
//        List<String> allLines = LoadFromFile(filename);
//        List<Double> xCoord = new ArrayList<>();
//        List<Double> yCoord = new ArrayList<>();
//        Random rnd = new Random();
//        int actualLine = 0;
//        for (String line : allLines) {
//            if (actualLine < startFrom - 1) {
//                actualLine += 1;
//                continue;
//            }
//            if (line != null) {
//                String[] split = line.trim().replaceAll("\\s{2,}", " ").split(" ");
//                if (split.length >= 3) {
//                    try {
//                        xCoord.add(Double.parseDouble(split[split.length - 2]));
//                        yCoord.add(Double.parseDouble(split[split.length - 1]));
//                        actualLine += 1;
//                    } catch (NumberFormatException nfe) {
//
//                    }
//                }
//            }
//        }
//        this.n = xCoord.size();
//        this.distances = new double[this.n][this.n];
//        for (int i = 0; i < n; i++) {
//            for (int j = 0; j < i; j++) {
//                distances[i][j] = Math.round(Math.sqrt(Math.pow(xCoord.get(i) - xCoord.get(j), 2) + Math.pow(yCoord.get(i) - yCoord.get(j), 2)))
//                        + rnd.nextDouble() * 0.002 - 0.001;
//                distances[j][i] = distances[i][j];
//            }
//            distances[i][i] = 0.0;
//        }
//    }

    public Distances(String filename, int number) {
        this.filename = filename;
        List<String> allLines = LoadFromFile(filename);
        this.n = number;
        this.distances = new double[this.n][this.n];
        int actualLine = 0;
        for (String line : allLines) {
            if (actualLine == 0) {
                actualLine = 1;
                continue;
            }
            if (line != null && actualLine < number + 1) {
                String[] split = line.split(";");
                for (int i = 1; i < split.length; i++) {
                    distances[i - 1][actualLine - 1] = Double.parseDouble(split[i]);
                }
                actualLine++;
            }
        }
    }

    public Distances(int n) {
        Random rndGen = new Random();
        this.n = n;
        this.distances = new double[this.n][this.n];
        this.filename = "Random n=" + n;
        Double[][] coord = new Double[2][n];
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < n; j++) {
                coord[i][j] = rndGen.nextDouble() * 100;
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                distances[i][j] = Math.sqrt(Math.pow(coord[0][i] - coord[0][j], 2) + Math.pow(coord[1][i] - coord[0][1], 2));
            }
        }
    }


    private static List<String> LoadFromFile(String filename) {
        BufferedReader reader;
        List<String> allLines = new ArrayList<>();
        try {
            reader = new BufferedReader(new FileReader(filename));
            String line = reader.readLine();
            allLines.add(line);

            while (line != null) {
                line = reader.readLine();
                allLines.add(line);
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return allLines;
    }
}
