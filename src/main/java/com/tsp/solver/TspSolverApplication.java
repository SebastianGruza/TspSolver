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
import java.util.stream.IntStream;

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
        Colony oldResults = new Colony();
        for (int epoch = 1; epoch < epochsInMain; epoch++) {
            System.out.println("Start epoch " + epoch);
            Instant startEpoch = Instant.now();
            copyPathsIntoOtherTable(ts, n, path, sum, sum3, path3);
            int[] isFaultIntegrity = new int[ts];
            TspGAKernel kernelGPU = new TspGAKernel(sum, path, gaResultSum, gaResult, distances, size, n, pm, epochsInGPU, isFaultIntegrity);
            //kernelGPU.setExecutionMode(Kernel.EXECUTION_MODE.JTP);
            kernelGPU.execute(Range.create(size));
            kernelGPU.dispose();
            checkIntegrityAndRepair(ts, n, path, sum, rndGen, isFaultIntegrity);
            Instant end = Instant.now();
            Duration timeElapsed = Duration.between(start, end);
            Duration timeElapsedEpoch = Duration.between(startEpoch, end);
            System.out.println("Time taken epochsInGPU = " + epochsInGPU + " on GPU: " + timeElapsedEpoch.toMillis() + " milliseconds,");
            System.out.println("Total: " + timeElapsed.toSeconds() + " seconds");
            System.out.println("End GPU calculation ");

            Integer threshold = (int) (ts / 8.0 / Math.log(epoch + 1));
            System.out.println("Threshold to repeat with actual data = " + threshold);

            Colony results = postEpochProcessing(ts, path, sum, epoch);
            copyPathsIntoOtherTable(ts, n, path, sum, sum2, path2);
            createNextGeneration(size, pm, ts, n, path, rndGen, results);
            if (oldResults.getIndividuals().size() + results.getIndividuals().size() < threshold) {
                copyPathsIntoOtherTable(ts, n, path3, sum3, sum, path);
                oldResults = new Colony(results, oldResults);
                System.out.println("--> Repeat calculation, unique = " + oldResults.getIndividuals().size());
            } else {
                oldResults = new Colony(results, oldResults);
                if (oldResults.getIndividuals().size() < threshold) {
                    copyPathsIntoOtherTable(ts, n, path3, sum3, sum, path);
                    System.out.println("--> Repeat calculation, unique = " + oldResults.getIndividuals().size());
                } else {
                    System.out.println("--> Calculation with unique " + oldResults.getIndividuals().size());
                    //prepare  table path base on results:
                    if (results.getIndividuals().size() < oldResults.getIndividuals().size()) {
                        System.out.println("--> Preparing table based all unique results");
                        copyPathsIntoOtherTable(ts, n, path, sum, sum2, path2);
                        createNextGeneration(size, pm, ts, n, path, rndGen, oldResults);
                    }
                    oldResults = new Colony();
                }
            }

            System.out.println("End CPU calculation");
        }
    }

    private void checkIntegrityAndRepair(int ts, int n, int[][] path, double[] sum, Random rndGen, int[] isFaultIntegrity) {
        Integer faultIntegrity = Arrays.stream(isFaultIntegrity).sum();
        if (faultIntegrity > 0) {
            System.out.println("----> ERROR CHECKING CORRECTION, integrity fault detected in " + faultIntegrity);
            for (int i = 0; i < isFaultIntegrity.length; i++) {
                if (isFaultIntegrity[i] > 0) {
                    Integer rnd;
                    do {
                        rnd = rndGen.nextInt(ts);
                    } while (isFaultIntegrity[rnd] > 0);
                    for (int j = 0; j < n; j++) {
                        path[i][j] = path[rnd][j];
                    }
                    sum[i] = sum[rnd];
                }
            }
        }
    }

    private static Colony postEpochProcessing(int ts, int[][] path, double[] sum, int epoch) {
        Map<Path, int[]> distinct = getDistinctPathWithIndex(ts, sum, path, epoch);
        int[] sortedIndices = IntStream.range(0, sum.length)
                .boxed().sorted((i, j) -> (sum[i] < sum[j]) ? -1 : (sum[i] > sum[j]) ? 1 : 0)
                .mapToInt(ele -> ele).toArray();

        System.out.println("Best    =  " + sum[sortedIndices[0]]);
        System.out.println("Worst   =  " + sum[sortedIndices[ts - 1]]);
        System.out.println("Unique  =  " + distinct.size());

        double best = sum[sortedIndices[0]];
        double worst = sum[sortedIndices[ts - 1]];
        Arrays.sort(sum);
        return new Colony(distinct, best, worst);
    }

    private static Map<Path, int[]> getDistinctPathWithIndex(int ts, double[] sum, int[][] path, int epoch) {
        Map<Path, int[]> distinct = new HashMap<>();
        double total = 0.0;
        for (int i = 0; i < ts; i++) {
            total += sum[i] / ts;
            distinct.put(new Path(sum[i]), path[i]);
        }
        System.out.println("Epoch   =  " + epoch);
        System.out.println("Mean    =  " + total);
        return distinct;
    }

    private static void createNextGeneration(int size, int pm, int ts, int n, int[][] path, Random rndGen, Colony colony) throws InterruptedException {
        List<Path> sequence = colony.getIndividuals().keySet().stream().sorted(Comparator.comparing(Path::getTotal)).collect(Collectors.toList());
        Double best = sequence.get(0).getTotal();
        Double worst = sequence.get(sequence.size() - 1).getTotal();
        List<Integer> listToParallel = new ArrayList<>(size);
        for (Integer j = 0; j < size; j++) {
            listToParallel.add(j);
        }

        Integer scalePower = 400;
        Double power = Math.log(((sequence.size() / ts + 1.0) * (worst / best) - 1) * scalePower + 1) + 1;
        System.out.println("Actual power = " + power);

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
                            if (seqSize < pm) {
                                System.out.println("WARNING: Unique size below than pm=" + pm);
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