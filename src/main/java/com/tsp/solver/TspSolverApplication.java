package com.tsp.solver;

import com.aparapi.Kernel;
import com.aparapi.Range;
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


        final int size = 1024; // 4096 active cores
        final int pm = 4; //path multiplier
        final int ts = size * pm; //total size
        Distances dist = new Distances("full.txd");
        final int n = dist.n;

        final double[][] distances = dist.distances;
        final int[][] path = new int[ts][n];
        final double[] sum = new double[ts];
        final int[][] path2 = new int[ts][n];
        final int[][] gaResult = new int[ts][n];
        final double[] gaResultSum = new double[ts];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < ts; j++) {
                path[j][i] = (i + j) % n;
            }
        }

        Instant start;
        System.out.println("START");
        GreedyAlgorithm.CreateNewGenerationWithGreedyAlgorithm(n/10, 1, distances, path, ts);
        System.out.println("GreedyAlgorithm check");
        Random rndGen = new Random();
        int epochsInGPU = 10;
        int epochsInMain = 100;
        for (int epoch = 1; epoch < epochsInMain; epoch++) {
            System.out.println("Start epoch " + epoch);
            start = Instant.now();
            TspGAKernel kernelGPU = new TspGAKernel(sum, path, gaResultSum, gaResult, distances, size, n, pm, epochsInGPU);
            kernelGPU.execute(Range.create(size));
            kernelGPU.dispose();
            Instant end = Instant.now();
            Duration timeElapsed = Duration.between(start, end);
            System.out.println("Time taken epochsInGPU = " + epochsInGPU + " on GPU: " + timeElapsed.toMillis() + " milliseconds");
            System.out.println("End GPU calculation");

            int epochsInCPU = epochsInGPU;
            start = Instant.now();
            TspGAKernel kernelCPU = new TspGAKernel(sum, path, gaResultSum, gaResult, distances, size, n, pm, epochsInCPU);
            kernelCPU.setExecutionMode(Kernel.EXECUTION_MODE.JTP);
            kernelCPU.execute(Range.create(size));
            kernelCPU.dispose();
            end = Instant.now();
            timeElapsed = Duration.between(start, end);
            System.out.println("Time taken epochsInCPU = " + epochsInCPU + " on CPU: " + timeElapsed.toMillis() + " milliseconds");


            processingAfterGPU(size, pm, ts, n, path, sum, path2, rndGen, epoch);
            System.out.println("End CPU calculation");
        }
    }

    private static void processingAfterGPU(int size, int pm, int ts, int n, int[][] path, double[] sum, int[][] path2, Random rndGen, int epoch) throws InterruptedException {
        Map<Path, Integer> distinct = new HashMap<>();

        double total = 0.0;
        for (int i = 0; i < ts; i++) {
            total += sum[i] / ts;
            distinct.put(new Path(sum[i]), i);
        }
        int[] sortedIndices = IntStream.range(0, sum.length)
                .boxed().sorted((i, j) -> (sum[i] < sum[j]) ? -1 : (sum[i] > sum[j]) ? 1 : 0)
                .mapToInt(ele -> ele).toArray();

        System.out.println("Mean in " + epoch + ": " + total);
        System.out.println("Best    =  " + sum[sortedIndices[0]]);
        System.out.println("Worst   =  " + sum[sortedIndices[ts - 1]]);
        System.out.println("Unique=  " + distinct.size());


        double best = sum[sortedIndices[0]];
        double worst = sum[sortedIndices[ts - 1]];
        Arrays.sort(sum);
        Map<Path, Double> probability = new HashMap<>();
        for (int i = 0; i < ts; i++) {
            Double actualWeight = Math.pow((worst - sum[i]) / (worst - best) + 0.5, 3);
            probability.put(new Path(sum[i]), actualWeight);
        }
        List<Path> sequence = probability.entrySet().stream().sorted((a, b) -> b.getValue().compareTo(a.getValue())).map(a -> a.getKey()).collect(Collectors.toList());

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < ts; j++) {
                path2[j][i] = path[j][i];
            }
        }
        List<Integer> listToParallel = new ArrayList<>(size);
        for (Integer j = 0; j < size; j++) {
            listToParallel.add(j);
        }

        ForkJoinPool newCustomThreadPool = new ForkJoinPool(24);
        try {
            newCustomThreadPool.submit(
                    () -> {
            listToParallel.parallelStream().forEach((j)->{
                List<Integer> selectorList = new ArrayList<>(pm);
                for (int k = 0; k < pm; k++) {
                    Integer seqSize = sequence.size();
                    Double toCalculate = rndGen.nextDouble();
                    int selector = (int) (Math.pow(toCalculate, 4) * (seqSize - 2));
                    selectorList.add(selector);
                }
                selectorList = selectorList.stream().sorted().distinct().collect(Collectors.toList());
                while (selectorList.size() < pm) {
                    Integer last = selectorList.get(selectorList.size() - 1);
                    if (last < sequence.size()) {
                        selectorList.add(last + 1);
                    } else {
                        selectorList.add(last - 1);
                    }
                }

                for (int i = 0; i < n; i++) {
                    for (int k = 0; k < pm; k++) {
                        path[4 * j + k][i] = path2[distinct.get(sequence.get(selectorList.get(k)))][i];
                    }
                }
            });}).get();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
    }
}