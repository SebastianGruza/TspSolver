package com.tsp.solver.data;

import com.tsp.solver.configuration.AppConfiguration;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Distances {
    private String filename;
    public int n;
    public double[][] distances;
    public double[] x;
    public double[] y;
    public String bestSolution = null;

    private static final double COORDINATE_SCALING_FACTOR = 0.002;
    private static final double COORDINATE_OFFSET = 0.001;

    public Distances(AppConfiguration tspProperties) {
        this.filename = tspProperties.getFilename();

        List<String> allLines = loadFromFile(filename);
        if (this.filename.endsWith(".txd")) {
            parseTxdFile(allLines);
        } else if (this.filename.endsWith(".tsp")) {
            parseTspFile(allLines);
        } else {
            throw new IllegalArgumentException("File format not supported");
        }
    }

    private void parseTxdFile(List<String> lines) {
        List<Double> xCoord = new ArrayList<>();
        List<Double> yCoord = new ArrayList<>();
        for (String line : lines) {
            if (line != null) {
                String[] split = line.split(";");
                if (split.length == 3) {
                    try {
                        xCoord.add(Double.parseDouble(split[1]));
                        yCoord.add(Double.parseDouble(split[2]));
                    } catch (NumberFormatException nfe) {
                        // Consider adding logging here
                    }
                }
            }
        }
        this.n = xCoord.size();
        this.distances = new double[this.n][this.n];
        this.x = xCoord.stream().mapToDouble(Double::doubleValue).toArray();
        this.y = yCoord.stream().mapToDouble(Double::doubleValue).toArray();
        calculateDistances(xCoord, yCoord);
    }

    private void parseTspFile(List<String> lines) {
        int startFrom = 7;
        List<Double> xCoord = new ArrayList<>();
        List<Double> yCoord = new ArrayList<>();
        Random rnd = new Random();
        int actualLine = 0;
        for (String line : lines) {
            if (actualLine < startFrom - 1) {
                actualLine++;
                continue;
            }
            if (line != null) {
                String[] split = line.trim().replaceAll("\\s{2,}", " ").split(" ");
                if (split.length >= 3) {
                    try {
                        xCoord.add(Double.parseDouble(split[split.length - 2]));
                        yCoord.add(Double.parseDouble(split[split.length - 1]));
                        actualLine++;
                    } catch (NumberFormatException nfe) {
                        // Consider adding logging here
                    }
                }
            }
        }
        this.n = xCoord.size();
        this.distances = new double[this.n][this.n];
        this.x = xCoord.stream().mapToDouble(Double::doubleValue).toArray();
        this.y = yCoord.stream().mapToDouble(Double::doubleValue).toArray();
        calculateDistancesTsp(xCoord, yCoord, rnd);
    }

    private void calculateDistances(List<Double> xCoord, List<Double> yCoord) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                distances[i][j] = Math.sqrt(Math.pow(xCoord.get(i) - xCoord.get(j), 2) + Math.pow(yCoord.get(i) - yCoord.get(j), 2));
            }
        }
    }

    private void calculateDistancesTsp(List<Double> xCoord, List<Double> yCoord, Random rnd) {
        for (int i = 0; i < n; i++) {
            System.out.println("i: " + i);
            for (int j = 0; j < i; j++) {
                distances[i][j] = Math.round(Math.sqrt(Math.pow(xCoord.get(i) - xCoord.get(j), 2) + Math.pow(yCoord.get(i) - yCoord.get(j), 2)))
                        + rnd.nextDouble() * COORDINATE_SCALING_FACTOR - COORDINATE_OFFSET;
                distances[j][i] = distances[i][j];
            }
            distances[i][i] = 0.0;
        }
    }

    private List<String> loadFromFile(String filename) {
        try {
            return Files.readAllLines(Paths.get(filename));
        } catch (IOException e) {
            // Consider adding logging or throwing a custom exception here
        }
        return Collections.emptyList();
    }

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
        this.x = new double[n];
        this.y = new double[n];
        for (int i = 0; i < n; i++) {
            this.x[i] = coord[0][i];
            this.y[i] = coord[1][i];
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
