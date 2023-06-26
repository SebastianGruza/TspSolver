package com.tsp.solver;

import com.aparapi.Range;
import com.tsp.solver.configuration.AppConfiguration;
import com.tsp.solver.data.Colony;
import com.tsp.solver.data.Distances;
import com.tsp.solver.data.Path;
import org.springframework.beans.factory.annotation.Autowired;
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

    @Autowired
    Distances dist;

    @Autowired
    AppConfiguration appConfiguration;
    public static void main(String[] args) {

        SpringApplication.run(TspSolverApplication.class, args);

    }

    @Override
    public void run(String... args) throws InterruptedException {

        Integer counterTotal = 0;
        while (true) {
            start();
            System.out.println();
            System.out.println("END counterTotal number " + counterTotal++);
            System.out.println();
        }
    }

    public void start() throws InterruptedException {

        final int size = appConfiguration.getGpuThreads();
        final int sizeTabu = 131072;
        final int pm = 4; //path multiplier - how many paths per thread
        final int ts = size * pm; //total size of paths
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
        Instant startEpoch = Instant.now();
        if (appConfiguration.getDivideGreedy() > 0) {
            GreedyAlgorithm.CreateNewGenerationWithGreedyAlgorithm(n / appConfiguration.getDivideGreedy(), 1, distances, path, ts);
        }
        System.out.println("GreedyAlgorithm check");
        Random rndGen = new Random();
        int epochsInGPU = 3;
        int epochsInMain = 100000;
        int colonyMultiplier = appConfiguration.getColonyMultiplier();
        int bestsHistoricalCounter = size / 8;
        List<Set<Path>> allPaths = new ArrayList<>();
        for (int i = 0; i < colonyMultiplier; i++) {
            allPaths.add(new HashSet<>());
        }
        Integer counterMerge = 0;
        Integer counterTotalMerge = 0;
        Boolean onlyMutate = true;
        List<Map<Path, int[]>> bestsHistorical = new ArrayList<>();
        Map<Path, Integer> countingToTabu = new HashMap<>();
        List<Colony> oldResults = new ArrayList<>(colonyMultiplier);
        for (int i = 0; i < colonyMultiplier; i++) {
            oldResults.add(new Colony());
            bestsHistorical.add(new HashMap<>());
        }


        for (int epoch = 1; epoch < epochsInMain; epoch++) {
            if (epoch > 10) {
                onlyMutate = false;
            }
            if (counterTotalMerge >= 100) {
                break;
            }
            System.out.println("Start epoch " + epoch);

            copyPathsIntoOtherTable(ts, n, path, sum, sum3, path3);
            int[] isFaultIntegrity = new int[ts];
            List<Double> tabuList = countingToTabu
                    .entrySet().parallelStream().filter(a -> a.getValue() > 8)
                    .sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                    .limit(sizeTabu)
                    .map(a -> a.getKey().getTotal()).collect(Collectors.toList());
            int countTabu = tabuList.size();
            if (countTabu > 0) {
                System.out.println("Tabu best path: " + tabuList.get(0));
            }
            int depthBst = countTabu == 0 ? 0 : 32 - Integer.numberOfLeadingZeros(countTabu - 1);
            depthBst = Math.max(1, depthBst);
            int sizeBst = (1 << depthBst) - 1;
            System.out.println("Tabu path total: " + countTabu);
            if (countTabu < sizeBst) {
                int elementsToAdd = sizeBst - countTabu;
                for (int i = 0; i < elementsToAdd; i++) {
                    tabuList.add(Double.MAX_VALUE);
                }
            }

            double bstTable[] = createBst(tabuList, depthBst, sizeBst);

            Integer trialsCrossover = onlyMutate ? 0 : 12;
            Instant end = Instant.now();
            Duration timeElapsedEpoch = Duration.between(startEpoch, end);
            System.out.println("Time taken preEpoch on CPU: " + timeElapsedEpoch.toMillis() + " milliseconds,");
            System.out.println("End CPU calculation ");
            startEpoch = Instant.now();
            TspGAKernel kernelGPU = new TspGAKernel(sum, path, gaResultSum, gaResult, distances, size, n, pm, epochsInGPU, isFaultIntegrity, trialsCrossover, trialsCrossover * 2, epoch, bstTable, depthBst);
            //kernelGPU.setExecutionMode(Kernel.EXECUTION_MODE.JTP);
            kernelGPU.execute(Range.create(size));
            kernelGPU.dispose();
            checkIntegrityAndRepair(ts, n, path, sum, rndGen, isFaultIntegrity, colonyMultiplier, pm);
            end = Instant.now();
            Duration timeElapsed = Duration.between(start, end);
            timeElapsedEpoch = Duration.between(startEpoch, end);
            System.out.println("Time taken epochsInGPU = " + epochsInGPU + " on GPU: " + timeElapsedEpoch.toMillis() + " milliseconds,");
            System.out.println("Total: " + timeElapsed.toSeconds() + " seconds");
            System.out.println("End GPU calculation ");
            startEpoch = Instant.now();
            List<Colony> results = postEpochProcessing(ts, path, sum, epoch, colonyMultiplier);

            List<Integer> distinct = results.stream().map(c -> c.getIndividuals().size()).collect(Collectors.toList());
            Integer distintSum = distinct.stream().mapToInt(a -> a).sum();
            System.out.println("Unique individuals of all colonies = " + distintSum);

            counterMerge++;
            if ((distintSum < ts / 32 && counterMerge > 32) || distintSum < ts / 64) {
                counterTotalMerge++;
                counterMerge = 0;
                System.out.println("------> MERGE last colonies now <------");
                onlyMutate = false;
                Colony total = new Colony();
                Integer bestId = 0;
                for (Colony colony : results) {
                    if (epoch > 20 && colony.getIndividuals().size() < bestsHistoricalCounter / 4) {
                        Map<Path, int[]> actualBest = bestsHistorical.get(bestId).entrySet().stream()
                                .sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                                .limit(bestsHistoricalCounter).collect(Collectors.toMap(a -> a.getKey(), a -> a.getValue()));
                        for (Path pathCandidate : colony.getIndividuals().keySet()) {
                            if (countingToTabu.containsKey(pathCandidate)) {
                                countingToTabu.put(pathCandidate, countingToTabu.get(pathCandidate) + 1);
                            } else {
                                countingToTabu.put(pathCandidate, 1);
                            }
                        }
                        colony.getIndividuals().putAll(actualBest);
                    }
                    total = new Colony(total, colony);
                }
                List<Colony> oneBigColony = new ArrayList<>();
                oneBigColony.add(total);
                copyPathsIntoOtherTable(ts, n, path, sum, sum2, path2);
                createNextGeneration(size, pm, ts, n, path, rndGen, oneBigColony, 40);
//                allPaths.addAll(oneBigColony.stream()
//                        .map(a -> a.getIndividuals().keySet())
//                        .flatMap(a -> a.stream())
//                        .collect(Collectors.toList()));
            } else {
                int i = 0;
                for (Colony colony : results) {
                    int j = 0;
                    for (Colony colony2 : results) {
                        Set<Path> intersection = new HashSet<>(colony.getIndividuals().keySet());
                        intersection.retainAll(colony2.getIndividuals().keySet());
                        Integer intersectionSum = intersection.size();
                        System.out.print(String.format(" %5d", intersectionSum));
//                        if (i < j && epoch % 10 == 5) {
//                            colony.setIndividuals(colony.getIndividuals()
//                                    .entrySet().stream().filter(a -> !intersection.contains(a.getKey()))
//                                    .collect(Collectors.toMap(a -> a.getKey(), a -> a.getValue())));
//                        }
                        j++;
                    }
                    i++;
                    System.out.println();
                }
                System.out.println("Intersection with all historical paths");
                i = 0;
                for (Colony colony : results) {
                    int j = 0;
                    for (Set<Path> colony2 : allPaths) {
                        Set<Path> intersection = new HashSet<>(colony.getIndividuals().keySet());
                        intersection.retainAll(colony2);
                        Integer intersectionSum = intersection.size();
                        System.out.print(String.format(" %5d", intersectionSum));
//                        if (i != j && epoch % 10 == 5) {
//                            colony.setIndividuals(colony.getIndividuals()
//                                    .entrySet().stream().filter(a -> !intersection.contains(a.getKey()))
//                                    .collect(Collectors.toMap(a -> a.getKey(), a -> a.getValue())));
//                        }
                        j++;
                    }

                    i++;
                    System.out.println();
                }
                //System.out.println("Intersect with existed all paths:");
                Integer bestId = 0;
                Map<Path, int[]> bests = new HashMap<>();
                for (Colony colony : results) {
//                    for (Set<Path> paths : allPaths) {
//                        Set<Path> intersection = new HashSet<>(colony.getIndividuals().keySet());
//                        intersection.retainAll(paths);
//                        Integer intersectionSum = intersection.size();
//                        System.out.print(String.format(" %5d", intersectionSum));
//                    }
//                    System.out.println();
                    bestsHistorical.get(bestId).putAll(colony.getIndividuals());
                    Map<Path, int[]> actualBest = bestsHistorical.get(bestId).entrySet().stream()
                            .sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                            .limit(bestsHistoricalCounter).collect(Collectors.toMap(a -> a.getKey(), a -> a.getValue()));
                    bestsHistorical.get(bestId).clear();
                    bestsHistorical.get(bestId).putAll(actualBest);
                    bestId++;
                    bests.putAll(actualBest.entrySet().stream().sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                            .limit(1).collect(Collectors.toMap(a->a.getKey(), a->a.getValue())));
                    if (epoch > 5 && colony.getIndividuals().size() < bestsHistoricalCounter / 4) {
                        for (Path pathCandidate : colony.getIndividuals().keySet()) {
                            if (countingToTabu.containsKey(pathCandidate)) {
                                countingToTabu.put(pathCandidate, countingToTabu.get(pathCandidate) + 1);
                            } else {
                                countingToTabu.put(pathCandidate, 1);
                            }
                        }
                        colony.getIndividuals().putAll(actualBest);
                    }
                }

                Map<Path, int[]> best = bests.entrySet().stream().sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                        .limit(1).collect(Collectors.toMap(a-> a.getKey(), a->a.getValue()));
                Path bestKey = best.keySet().iterator().next();
                System.out.println("Best path = " + bestKey);
                String bestSolution = "";
                for (int j = 0; j < best.get(bestKey).length; j++) {
                    System.out.print("-" + best.get(bestKey)[j]);
                    bestSolution += "-" + best.get(bestKey)[j];
                };
                bestSolution = bestSolution.substring(1);
                dist.bestSolution = bestSolution;


                System.out.println();
                copyPathsIntoOtherTable(ts, n, path, sum, sum2, path2);
                createNextGeneration(size, pm, ts, n, path, rndGen, results, 400);
                int colonyNumber = 0;
                for (Set<Path> paths : allPaths) {
                    paths.addAll(results.get(colonyNumber).getIndividuals().keySet());
                    colonyNumber++;
                }
            }
            countingToTabu = countingToTabu
                    .entrySet().parallelStream()
                    .sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                    .limit(524288).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
            //System.out.println("Total unique paths in algorithm = " + allPaths.size());
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

    private double[] createBst(List<Double> list, int depthBst, int sizeBst) {
        double[] bstTable = new double[sizeBst];
        int base = sizeBst;
        int start = 0;
        for (int level = 0; level < depthBst; level++) {
            int elementsInLevel = 1 << level;
            int middle = base / 2;
            base = base / 2;
            start += elementsInLevel / 2;
            for (int j = start; j < start + elementsInLevel; j++) {
                bstTable[j] = list.get(middle);
                middle += 1 << (depthBst - level);
            }
        }
        return bstTable;
    }

    private int searchInBst(double value, double[] bstTable, int depthBst) {
        int startId = 0;
        for (int level = 0; level < depthBst; level++) {
            double vertexInBst = bstTable[startId];
            int leftId = (startId + 1) * 2 - 1;
            int rightId = (startId + 1) * 2;
            if (vertexInBst <= value + 0.0000001 &&
                    vertexInBst >= value - 0.0000001) {
                return 1;
            } else if (vertexInBst > value && leftId < 1 << (depthBst)) {
                startId = leftId;
            } else if (vertexInBst < value && rightId < 1 << (depthBst)) {
                startId = rightId;
            } else {
                return 0;
            }
        }
        return 0;
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
        Map<Double, Map<Path, int[]>> sortedMap = new HashMap<>();
        Map<Double, Path> bestPaths = new HashMap<>();
        Map<Double, Path> worstPaths = new HashMap<>();
        Integer howMany = ts / colonyMultiplier;
        Map<Double, String> toPrint = new HashMap<>();
        for (int col = 0; col < colonyMultiplier; col++) {
            Integer start = col * howMany;
            Integer end = (col + 1) * howMany;
            Map<Path, int[]> distinct = new HashMap<>();
            double mean = 0.0;
            for (int i = start; i < end; i++) {
                mean += sum[i] / ts * colonyMultiplier;
                distinct.put(new Path(sum[i]), path[i]);
            }
            ret.add(distinct);
            List<Path> sequence = distinct.keySet().stream().sorted(Comparator.comparing(Path::getTotal)).collect(Collectors.toList());
            Double best = sequence.get(0).getTotal();
            Double worst = sequence.get(sequence.size() - 1).getTotal();
            Path bestPath = sequence.get(0);
            Path worstPath = sequence.get(sequence.size() - 1);
            String out = String.format("Colony = %3d", col);
            out += String.format(", Epoch = %3d", epoch);
            out += String.format(", Mean = %.3f", mean);
            out += String.format(", Best = %.6f", best);
            out += String.format(", Worst = %.6f", worst);
            out += String.format(", Unique = %d", distinct.size());
            toPrint.put(mean, out);
            sortedMap.put(mean, distinct);
            bestPaths.put(mean, bestPath);
            worstPaths.put(mean, worstPath);
        }
        Map<Path, int[]> previous = null;
        Path previousWorstPath = null;
        for (Double mean : toPrint.keySet().stream().sorted().collect(Collectors.toList())) {
            System.out.println(toPrint.get(mean));
            if (previousWorstPath != null && previous != null) {
                Path actualBest = bestPaths.get(mean);
//                if (previousWorstPath.getTotal() < actualBest.getTotal()) {
//                    Map<Path, int[]> actualDistinct = sortedMap.get(mean);
//                    actualDistinct.put(previousWorstPath, previous.get(previousWorstPath));
//                }
            }
            previousWorstPath = worstPaths.get(mean);
            previous = sortedMap.get(mean);
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