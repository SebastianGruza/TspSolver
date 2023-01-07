package com.tsp.solver;

import com.aparapi.Range;
import com.tsp.solver.data.Colony;
import com.tsp.solver.data.Distances;
import com.tsp.solver.data.Path;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;

@SpringBootApplication
public class TspSolverApplication implements CommandLineRunner {

    public static void main(String[] args) {

        SpringApplication.run(TspSolverApplication.class, args);

    }

    @Override
    public void run(String... args) throws InterruptedException {


        final int size = 16384; // 4096 active cores
        final int pm = 4; //path multiplier
        final int ts = size * pm; //total size
        Distances dist = new Distances("full.txd");
        final int n = dist.n;

        final double[][] distances = dist.distances;
        final int[][] path = new int[ts][n];
        final double[] sum = new double[ts];
        final int[][] path2 = new int[ts][n];
        final double[] sum2 = new double[ts];
        final int[][] path3 = new int[ts][n];
        final double[] sum3 = new double[ts];
        final int[][] gaResult = new int[ts][n];
        final double[] gaResultSum = new double[ts];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < ts; j++) {
                path[j][i] = (i + j) % n;
            }
        }

        System.out.println("START");
        Instant start = Instant.now();
        GreedyAlgorithm.CreateNewGenerationWithGreedyAlgorithm(n, 1, distances, path, ts);
        System.out.println("GreedyAlgorithm check");
        Random rndGen = new Random();
        int epochsInGPU = 10;
        int epochsInMain = 100000;
        int colonyMultiplier = 16;
        int bestsHistoricalCounter = 512;
        Set<Path> allPaths = new HashSet<>();
        Integer counterMerge = 0;
        List<Map<Path, int[]>> bestsHistorical = new ArrayList<>();
        List<Colony> oldResults = new ArrayList<>(colonyMultiplier);
        for (int i = 0; i < colonyMultiplier; i++) {
            oldResults.add(new Colony());
            bestsHistorical.add(new HashMap<>());
        }


        for (int epoch = 1; epoch < epochsInMain; epoch++) {
            System.out.println("Start epoch " + epoch);
            Instant startEpoch = Instant.now();
            copyPathsIntoOtherTable(ts, n, path, sum, sum3, path3);
            int[] isFaultIntegrity = new int[ts];
            TspGAKernel kernelGPU = new TspGAKernel(sum, path, gaResultSum, gaResult, distances, size, n, pm, epochsInGPU, isFaultIntegrity);
            //kernelGPU.setExecutionMode(Kernel.EXECUTION_MODE.JTP);
            kernelGPU.execute(Range.create(size));
            kernelGPU.dispose();
            checkIntegrityAndRepair(ts, n, path, sum, rndGen, isFaultIntegrity, colonyMultiplier, pm);
            Instant end = Instant.now();
            Duration timeElapsed = Duration.between(start, end);
            Duration timeElapsedEpoch = Duration.between(startEpoch, end);
            System.out.println("Time taken epochsInGPU = " + epochsInGPU + " on GPU: " + timeElapsedEpoch.toMillis() + " milliseconds,");
            System.out.println("Total: " + timeElapsed.toSeconds() + " seconds");
            System.out.println("End GPU calculation ");
            List<Colony> results = postEpochProcessing(ts, path, sum, epoch, colonyMultiplier);

            List<Integer> distinct = results.stream().map(c -> c.getIndividuals().size()).collect(Collectors.toList());
            Integer distintSum = distinct.stream().mapToInt(a -> a).sum();
            System.out.println("Unique individuals of all colonies = " + distintSum);

            counterMerge++;
            if ((distintSum < ts / 32 && counterMerge > 32) || distintSum < ts / 64) {
                counterMerge = 0;
                System.out.println("------> MERGE last colonies now <------");
                Colony total = new Colony();
                Integer bestId = 0;
                for (Colony colony : results) {
                    if (epoch > 20 && colony.getIndividuals().size() < bestsHistoricalCounter / 4) {
                        Map<Path, int[]> actualBest = bestsHistorical.get(bestId).entrySet().stream()
                                .sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                                .limit(bestsHistoricalCounter).collect(Collectors.toMap(a -> a.getKey(), a -> a.getValue()));
                        colony.getIndividuals().putAll(actualBest);
                    }
                    total = new Colony(total, colony);
                }
                List<Colony> oneBigColony = new ArrayList<>();
                oneBigColony.add(total);
                copyPathsIntoOtherTable(ts, n, path, sum, sum2, path2);
                createNextGeneration(size, pm, ts, n, path, rndGen, oneBigColony, 40);
                allPaths.addAll(oneBigColony.stream()
                        .map(a -> a.getIndividuals().keySet())
                        .flatMap(a -> a.stream())
                        .collect(Collectors.toList()));
            } else {
                int i = 0;
                for (Colony colony : results) {
                    int j = 0;
                    for (Colony colony2 : results) {
                        Set<Path> intersection = new HashSet<>(colony.getIndividuals().keySet());
                        intersection.retainAll(colony2.getIndividuals().keySet());
                        Integer intersectionSum = intersection.size();
                        System.out.print(String.format(" %5d", intersectionSum));
                        if (i < j) {
                            colony.setIndividuals(colony.getIndividuals()
                                    .entrySet().stream().filter(a -> !intersection.contains(a.getKey()))
                                    .collect(Collectors.toMap(a -> a.getKey(), a -> a.getValue())));
                        }
                        j++;
                    }
                    i++;
                    System.out.println();
                }
                System.out.println("Intersect with existed all paths:");
                Integer bestId = 0;
                for (Colony colony : results) {
                    Set<Path> intersection = new HashSet<>(colony.getIndividuals().keySet());
                    intersection.retainAll(allPaths);
                    Integer intersectionSum = intersection.size();
                    System.out.print(String.format(" %5d", intersectionSum));
                    bestsHistorical.get(bestId).putAll(colony.getIndividuals());
                    Map<Path, int[]> actualBest = bestsHistorical.get(bestId).entrySet().stream()
                            .sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                            .limit(bestsHistoricalCounter).collect(Collectors.toMap(a -> a.getKey(), a -> a.getValue()));
                    bestsHistorical.get(bestId).clear();
                    bestsHistorical.get(bestId).putAll(actualBest);
                    bestId++;
                    if (epoch > 20 && colony.getIndividuals().size() < bestsHistoricalCounter / 4) {
                        colony.getIndividuals().putAll(actualBest);
                    }
                }

                System.out.println();
                copyPathsIntoOtherTable(ts, n, path, sum, sum2, path2);
                createNextGeneration(size, pm, ts, n, path, rndGen, results, 400);
                allPaths.addAll(results.stream()
                        .map(a -> a.getIndividuals().keySet())
                        .flatMap(a -> a.stream())
                        .collect(Collectors.toList()));
            }
            System.out.println("Total unique paths in algorithm = " + allPaths.size());
            System.out.println("End CPU calculation");
        }
    }

    private void checkIntegrityAndRepair(int ts, int n, int[][] path, double[] sum, Random rndGen, int[] isFaultIntegrity, int colonyMultiplier, int pm) {
        Integer faultIntegrity = Arrays.stream(isFaultIntegrity).sum();
        if (faultIntegrity > 0) {
            System.out.println("----> ERROR CHECKING CORRECTION, integrity fault detected in " + faultIntegrity);
            for (int i = 0; i < isFaultIntegrity.length; i++) {
                if (isFaultIntegrity[i] > 0) {
                    Integer rnd;
                    Integer part = i / (ts / colonyMultiplier);
                    Integer bound1 = part * (ts / colonyMultiplier);
                    Integer bound2 = (part + 1) * (ts / colonyMultiplier);
                    do {
                        rnd = rndGen.nextInt(bound2 - bound1) + bound1;
                    } while (isFaultIntegrity[rnd] > 0);
                    for (int j = 0; j < n; j++) {
                        path[i][j] = path[rnd][j];
                    }
                    sum[i] = sum[rnd];
                }
            }
        }
    }

    private static List<Colony> postEpochProcessing(int ts, int[][] path, double[] sum, int epoch, int colonyMultiplier) {
        List<Map<Path, int[]>> distinct = getDistinctPathWithIndex(ts, sum, path, epoch, colonyMultiplier);
        List<Colony> ret = new ArrayList<>(distinct.size());
        for (Map<Path, int[]> map : distinct) {

            List<Path> sequence = map.keySet().stream().sorted(Comparator.comparing(Path::getTotal)).collect(Collectors.toList());
            Double best = sequence.get(0).getTotal();
            Double worst = sequence.get(sequence.size() - 1).getTotal();
            ret.add(new Colony(map, best, worst));
        }
        return ret;
    }

    private static List<Map<Path, int[]>> getDistinctPathWithIndex(int ts, double[] sum, int[][] path, int epoch, int colonyMultiplier) {
        List<Map<Path, int[]>> ret = new ArrayList<>();
        Integer howMany = ts / colonyMultiplier;
        Map<Double, String> toPrint = new HashMap<>();
        for (int col = 0; col < colonyMultiplier; col++) {
            Integer start = col * howMany;
            Integer end = (col + 1) * howMany;
            Map<Path, int[]> distinct = new HashMap<>();
            double total = 0.0;
            for (int i = start; i < end; i++) {
                total += sum[i] / ts * colonyMultiplier;
                distinct.put(new Path(sum[i]), path[i]);
            }
            ret.add(distinct);
            List<Path> sequence = distinct.keySet().stream().sorted(Comparator.comparing(Path::getTotal)).collect(Collectors.toList());
            Double best = sequence.get(0).getTotal();
            Double worst = sequence.get(sequence.size() - 1).getTotal();
            String out = String.format("Colony = %3d", col);
            out+= String.format(", Epoch = %3d", epoch);
            out+= String.format(", Mean = %.3f", total);
            out+= String.format(", Best = %.6f", best);
            out+= String.format(", Worst = %.6f", worst);
            out+= String.format(", Unique = %d", distinct.size());
            toPrint.put(best, out);
        }
        for (Double best : toPrint.keySet().stream().sorted().collect(Collectors.toList())) {
            System.out.println(toPrint.get(best));
        }
        return ret;
    }

    private static void createNextGeneration(int size, int pm, int ts, int n, int[][] path, Random rndGen, List<Colony> severalColonies, Integer scalePower) throws InterruptedException {
        Integer numberOfColony = 0;
        Integer totalColonies = severalColonies.size();
        Integer howMany = size / totalColonies;
        for (Colony colony : severalColonies) {
            Integer start = numberOfColony * howMany;
            Integer end = (numberOfColony + 1) * howMany;
            numberOfColony++;
            List<Path> sequence = colony.getIndividuals().keySet().stream().sorted(Comparator.comparing(Path::getTotal)).collect(Collectors.toList());
            Double best = sequence.get(0).getTotal();
            Double worst = sequence.get(sequence.size() - 1).getTotal();
            List<Integer> listToParallel = new ArrayList<>(size);
            for (Integer j = start; j < end; j++) {
                listToParallel.add(j);
            }


            Double power = Math.log(((sequence.size() / ts + 1.0) * (worst / best) - 1) * scalePower + 1) + 1;

            ForkJoinPool newCustomThreadPool = new ForkJoinPool(24);
            try {
                newCustomThreadPool.submit(
                        () -> {
                            listToParallel.parallelStream().forEach((j) -> {
                                List<Integer> selectorList = new ArrayList<>(pm);
                                Integer seqSize = sequence.size();
                                for (int k = 0; k < pm; k++) {
                                    Double toCalculate = rndGen.nextDouble();
                                    int selector = (int) (Math.pow(toCalculate, power) * (seqSize - 2));
                                    selectorList.add(selector);
                                }
                                selectorList = selectorList.stream().sorted().distinct().collect(Collectors.toList());
                                if (seqSize <= pm + 1) {
                                    while (selectorList.size() < pm) {
                                        selectorList.add(0);
                                    }
                                }
                                while (selectorList.size() < pm) {
                                    Double toCalculate = rndGen.nextDouble();
                                    int selector = (int) (Math.pow(toCalculate, power) * (seqSize - 2));
                                    selectorList.add(selector);
                                    selectorList = selectorList.stream().sorted().distinct().collect(Collectors.toList());
                                }
                                for (int k = 0; k < pm; k++) {
                                    int[] onePath = colony.getIndividuals().get(sequence.get(selectorList.get(k)));
                                    for (int i = 0; i < n; i++) {
                                        path[4 * j + k][i] = onePath[i];
                                    }
                                }
                            });
                        }).get();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        }
    }

    private static void copyPathsIntoOtherTable(int ts, int n, int[][] path, double[] sum, double[] sum2, int[][] path2) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < ts; j++) {
                path2[j][i] = path[j][i];
            }
        }
        for (int j = 0; j < ts; j++) {
            sum2[j] = sum[j];
        }
    }
}