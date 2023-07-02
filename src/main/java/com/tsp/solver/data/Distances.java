package com.tsp.solver.data;

import com.tsp.solver.clustering.DBSCAN;
import com.tsp.solver.clustering.DistancesBetweenClusters;
import com.tsp.solver.configuration.AppConfiguration;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class Distances {
    private String filename;
    public int n;
    public double[][] distances;
    public double[] x;
    public double[] y;
    public String bestSolution = null;

    public List<Set<Integer>> clusters = null;

    private static final double COORDINATE_SCALING_FACTOR = 0.002;
    private static final double COORDINATE_OFFSET = 0.001;

    public Distances(AppConfiguration tspProperties) {
        loadDistances(tspProperties.getFilename());
    }

    public Distances(String filename) {
        loadDistances(filename);
    }

    private void loadDistances(String filename) {
        this.filename = filename;
        List<String> allLines = loadFromFile(filename);
        if (this.filename.endsWith(".txd")) {
            parseTxdFile(allLines);
        } else if (this.filename.endsWith(".tsp")) {
            parseTspFile(allLines);
        } else {
            throw new IllegalArgumentException("File format not supported");
        }
        getClusters();
    }

    private void getClusters() {
        DistancesBetweenClusters distancesBetweenClusters = new DistancesBetweenClusters();
        double[] kDistances = distancesBetweenClusters.kthNearestNeighborDistances(this.distances, 6);
        Arrays.sort(kDistances);
        Double meanKdistance = Arrays.stream(kDistances).average().getAsDouble();
        Double eps = meanKdistance * 2.5;
        DBSCAN dbscan = new DBSCAN(this.distances, eps, 1);
        clusters = dbscan.apply();
        System.out.println("Groups: " + clusters.size());
        System.out.println("Total points in clusters: " + clusters.stream().mapToInt(Set::size).sum());
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
        int lineWithEdgeWeightType = 5;
        String edgeWeightType = "NOT_SUPPORTED";
        List<Double> xCoord = new ArrayList<>();
        List<Double> yCoord = new ArrayList<>();
        Random rnd = new Random();
        int actualLine = 0;
        for (String line : lines) {
            if (actualLine < startFrom - 1) {
                if (line.startsWith("EDGE_WEIGHT_TYPE")) {
                    if (line.contains("GEO")) {
                        edgeWeightType = "GEO";
                    } else if (line.contains("ATT")) {
                        edgeWeightType = "ATT";
                    } else if (line.contains("EUC_2D")) {
                        edgeWeightType = "EUC_2D";
                    }
                }
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
        this.y = xCoord.stream().mapToDouble(Double::doubleValue).toArray();
        this.x = yCoord.stream().mapToDouble(Double::doubleValue).toArray();
        if (edgeWeightType.equals("GEO")) {
            distancesInGEO(xCoord, yCoord, rnd);
        } else if (edgeWeightType.equals("ATT")) {
            attDistance(xCoord, yCoord, rnd);
        } else if (edgeWeightType.equals("EUC_2D")) {
            calculateDistancesTsp(xCoord, yCoord, rnd);
        } else {
            throw new IllegalArgumentException("Edge weight type not supported");
        }
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

    private void attDistance(List<Double> xCoord, List<Double> yCord, Random rnd) {
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double x1 = xCoord.get(i);
                double x2 = xCoord.get(j);
                double y1 = yCord.get(i);
                double y2 = yCord.get(j);
                double xd = x1 - x2;
                double yd = y1 - y2;
                double rij = Math.sqrt((xd * xd + yd * yd) / 10.0);
                double tij = Math.round(rij);
                if (tij < rij)
                    distances[i][j] = tij + 1
                            + rnd.nextDouble() * COORDINATE_SCALING_FACTOR - COORDINATE_OFFSET;
                else
                    distances[i][j] = tij
                            + rnd.nextDouble() * COORDINATE_SCALING_FACTOR - COORDINATE_OFFSET;
                distances[j][i] = distances[i][j];
            }
            distances[i][i] = 0.0;
        }
    }

    private void distancesInGEO_Wrong(List<Double> xCoord, List<Double> yCoord, Random rnd) {
        int n = xCoord.size();
        double PI = Math.PI;
        double RRR = 6378.388; //6427.5
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double latitude1 = xCoord.get(i) * PI / 180.0;
                double longitude1 = yCoord.get(i) * PI / 180.0;
                double latitude2 = xCoord.get(j) * PI / 180.0;
                double longitude2 = yCoord.get(j) * PI / 180.0;

                double q1 = Math.cos(longitude1 - longitude2);
                double q2 = Math.cos(latitude1 - latitude2);
                double q3 = Math.cos(latitude1 + latitude2);
                distances[i][j] = Math.round(RRR * Math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
                        + rnd.nextDouble() * COORDINATE_SCALING_FACTOR - COORDINATE_OFFSET;
                distances[j][i] = distances[i][j];
            }
            distances[i][i] = 0.0;
        }
    }
    public void distancesInGEO(List<Double> xCoord, List<Double> yCoord, Random rnd) {
        int dim = xCoord.size();
        double[] latitude = new double[dim];
        double[] longitude = new double[dim];

        final double PI = 3.141592;
        for (int i = 0; i < dim; i++) {
            int deg = (int) (xCoord.get(i)).doubleValue();
            double min = xCoord.get(i) - deg;
            latitude[i] = PI * (deg + 5 * min / 3) / 180;
            deg = (int) yCoord.get(i).doubleValue();
            min = yCoord.get(i) - deg;
            longitude[i] = PI * (deg + 5 * min / 3) / 180;
        }

        final double RRR = 6378.388;
        for (int i = 0; i < dim; i++) {
            for (int j = i + 1; j < dim; j++) {
                double q1 = Math.cos(longitude[i] - longitude[j]);
                double q2 = Math.cos(latitude[i] - latitude[j]);
                double q3 = Math.cos(latitude[i] + latitude[j]);
                distances[i][j] = (int) (RRR * Math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
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
