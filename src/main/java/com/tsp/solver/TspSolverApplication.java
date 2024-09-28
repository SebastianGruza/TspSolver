package com.tsp.solver;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import com.aparapi.internal.kernel.KernelManager;
import com.tsp.solver.configuration.AppConfiguration;
import com.tsp.solver.data.Colony;
import com.tsp.solver.data.Distances;
import com.tsp.solver.data.DistancesService;
import com.tsp.solver.data.Path;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@SpringBootApplication
public class TspSolverApplication implements CommandLineRunner {

    @Autowired
    DistancesService distancesService;

    Distances distancesData;

    @Autowired
    AppConfiguration appConfiguration;

    int[][] paths;
    double[] pathTotals;
    int[][] pathsCopy;
    double[] pathTotalsCopy;
    int[][] pathsAux;
    double[] pathTotalsAux;
    int[][] gaResultPaths;
    double[] gaResultTotals;
    int numberOfCities;

    boolean onlyMutate = true;
    int epochsInGPU = 20;
    int gpuThreads;
    int pathsPerThread; // How many paths per thread
    int totalPaths; // Total number of paths

    public static void main(String[] args) {
        SpringApplication.run(TspSolverApplication.class, args);
    }

    @Override
    public void run(String... args) throws InterruptedException {
        int counterTotal = 0;
        String filename = appConfiguration.getFilename();
        distancesData = distancesService.getCurrentDistances();
        boolean isOKLastFile = true;

        while (true) {
            double scaleSeconds = appConfiguration.getScaleTime(); // 0.01 - is optimum
            int secondsCalculation = getSecondsCalculation(scaleSeconds);

            try {
                if (!isOKLastFile) {
                    filename = getNextTspFileFromActiveDirectory(filename);
                    distancesData = new Distances(filename);
                    distancesService.updateDistances(distancesData);
                    secondsCalculation = getSecondsCalculation(scaleSeconds);
                    isOKLastFile = true;
                }
                System.out.println("START with secondsCalculation " + secondsCalculation + " for " + filename);
                String result = startAndGetBest(secondsCalculation);
                PrintWriter out = new PrintWriter(new FileOutputStream(new File("results.txt"), true));
                out.println("TOTAL RESULT in " + secondsCalculation + " seconds for " + filename + " : " + result);
                out.close();
                System.out.println("TOTAL RESULT in " + secondsCalculation + " seconds for " + filename + " : " + result);
                System.out.println("END counterTotal number " + counterTotal++);
                System.out.println();
                filename = getNextTspFileFromActiveDirectory(filename);
                distancesData = new Distances(filename);
                distancesService.updateDistances(distancesData);
            } catch (IllegalArgumentException e) {
                isOKLastFile = false;
                System.out.println("File " + filename + " is not OK. EDGE Type is not supported.");
                e.printStackTrace();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private int getSecondsCalculation(double scaleSeconds) {
        return (int) (distancesData.n * Math.pow(Math.log10(distancesData.n), 4) * scaleSeconds) + 5;
    }

    private String getNextTspFileFromActiveDirectory(String lastFilename) {
        try (Stream<java.nio.file.Path> paths = Files.list(Paths.get("."))) {
            List<String> files = paths
                    .filter(Files::isRegularFile)
                    .filter(path -> path.getFileName().toString().endsWith(".tsp"))
                    .map(path -> path.getFileName().toString())
                    .sorted()
                    .collect(Collectors.toList());

            int index = files.indexOf(lastFilename);
            if (index >= 0 && index < files.size() - 1) {
                return files.get(index + 1);
            } else {
                return files.get(0);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public String startAndGetBest(int secondsCalculation) throws InterruptedException {
        numberOfCities = distancesData.n;
        final double[][] distances = distancesData.distances;
        gpuThreads = appConfiguration.getGpuThreads();
        final int tabuListSize = 131072;

        if (numberOfCities < 1000) {
            enableSecondPhase();
        } else {
            pathsPerThread = 1;
            totalPaths = gpuThreads * pathsPerThread;
            initializePaths();
            onlyMutate = true;
            epochsInGPU = 20;
        }
        int mergeFinishedCount = 0;

        initializePopulation();

        System.out.println("START");
        Instant startTime = Instant.now();
        Instant epochStartTime = Instant.now();

        if (appConfiguration.getDivideGreedy() > 0) {
            GreedyAlgorithm.createNewGenerationWithGreedyAlgorithm(numberOfCities / appConfiguration.getDivideGreedy(), 16, distances, paths, totalPaths);
        }
        System.out.println("GreedyAlgorithm check");

        Random randomGenerator = new Random();
        int maxEpochs = 100000;
        int colonyMultiplier = appConfiguration.getColonyMultiplier();
        int bestsHistoricalCounter = gpuThreads / 8;
        List<Set<Path>> allPaths = new ArrayList<>();
        for (int i = 0; i < colonyMultiplier; i++) {
            allPaths.add(new HashSet<>());
        }
        int mergeCounter = 0;
        int totalMergeCounter = 0;
        List<Map<Path, int[]>> bestsHistorical = new ArrayList<>();
        Map<Path, Integer> countingToTabu = new HashMap<>();
        List<Colony> previousResults = new ArrayList<>(colonyMultiplier);
        for (int i = 0; i < colonyMultiplier; i++) {
            previousResults.add(new Colony());
            bestsHistorical.add(new HashMap<>());
        }
        String returnResult = "";

        for (int epoch = 1; epoch < maxEpochs; epoch++) {
            if (totalMergeCounter >= 100) {
                break;
            }
            System.out.println("Start epoch " + epoch);

            copyPaths(totalPaths, numberOfCities, paths, pathTotals, pathsAux, pathTotalsAux);
            int[] integrityFaults = new int[totalPaths];

            List<Double> tabuList = countingToTabu.entrySet().parallelStream()
                    .filter(a -> a.getValue() > 5)
                    .sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                    .limit(tabuListSize)
                    .map(a -> a.getKey().getTotal())
                    .collect(Collectors.toList());
            int tabuCount = tabuList.size();
            if (tabuCount > 0) {
                System.out.println("Tabu best path: " + tabuList.get(0));
            }
            int bstDepth = tabuCount == 0 ? 0 : 32 - Integer.numberOfLeadingZeros(tabuCount - 1);
            bstDepth = Math.max(1, bstDepth);
            int bstSize = (1 << bstDepth) - 1;
            System.out.println("Tabu path total: " + tabuCount);
            if (tabuCount < bstSize) {
                int elementsToAdd = bstSize - tabuCount;
                for (int i = 0; i < elementsToAdd; i++) {
                    tabuList.add(Double.valueOf(Double.MAX_VALUE));
                }
            }

            double[] bstTable = createBst(tabuList, bstDepth, bstSize);

            int crossoverTrials = onlyMutate ? 0 : 12;
            Instant currentTime = Instant.now();
            Duration epochDuration = Duration.between(epochStartTime, currentTime);
            System.out.println("Time taken preEpoch on CPU: " + epochDuration.toMillis() + " milliseconds,");
            System.out.println("End CPU calculation");
            epochStartTime = Instant.now();

            List<OpenCLDevice> gpuDevices = OpenCLDevice.listDevices(Device.TYPE.GPU);

            if (gpuDevices.size() > 0) {
                // Wyświetl dostępne GPU
                for (int i = 0; i < gpuDevices.size(); i++) {
                    System.out.println("GPU Device " + i + ": " + gpuDevices.get(i).getShortDescription());
                }

                // Wybierz GPU o określonym indeksie
                int deviceIndex = 0; // Zmień na 1, jeśli chcesz użyć drugiego GPU
                OpenCLDevice selectedDevice = gpuDevices.get(deviceIndex);
                System.out.println("Wybrano GPU: " + selectedDevice.getShortDescription());

                // Ustaw preferowane urządzenia globalnie
                LinkedHashSet<Device> preferredDevices = new LinkedHashSet<>();
                preferredDevices.add(selectedDevice);
                KernelManager.instance().setDefaultPreferredDevices(preferredDevices);
                TspGAKernel kernelGPU = new TspGAKernel(pathTotals, paths, gaResultTotals, gaResultPaths, distances, gpuThreads, numberOfCities, pathsPerThread, epochsInGPU, integrityFaults, crossoverTrials, crossoverTrials * 2, epoch, bstTable, bstDepth);

                // Ustaw tryb wykonania na GPU
                kernelGPU.setExecutionMode(Kernel.EXECUTION_MODE.GPU);

                // Wykonaj kernel
                kernelGPU.execute(Range.create(gpuThreads));
                kernelGPU.dispose();
            } else {
                throw new RuntimeException("Nie znaleziono żadnych urządzeń GPU.");
            }

            checkAndRepairIntegrity(totalPaths, numberOfCities, paths, pathTotals, randomGenerator, integrityFaults, colonyMultiplier, pathsPerThread);

            currentTime = Instant.now();
            Duration totalDuration = Duration.between(startTime, currentTime);
            epochDuration = Duration.between(epochStartTime, currentTime);
            System.out.println("Time taken epochsInGPU = " + epochsInGPU + " on GPU: " + epochDuration.toMillis() + " milliseconds,");
            System.out.println("Total: " + totalDuration.toSeconds() + " seconds");
            System.out.println("End GPU calculation");
            epochStartTime = Instant.now();

            List<Colony> results = postEpochProcessing(totalPaths, paths, pathTotals, epoch, colonyMultiplier);

            List<Integer> uniqueCounts = results.stream().map(c -> (Integer) c.getIndividuals().size()).collect(Collectors.toList());
            int totalUnique = uniqueCounts.stream().mapToInt(a -> a).sum();
            System.out.println("Unique individuals of all colonies = " + totalUnique);

            mergeCounter++;
            double mergeTimeRatio = Duration.between(startTime, Instant.now()).toSeconds() * 1.0 / secondsCalculation;
            if (shouldMergeColonies(totalPaths, mergeCounter, totalUnique, appConfiguration.getMergeColonyByTime(), appConfiguration.getCutoffsByTime(), mergeTimeRatio, mergeFinishedCount) && !onlyMutate) {
                mergeFinishedCount++;
                totalMergeCounter++;
                mergeCounter = 0;
                System.out.println("------> MERGE last colonies now <------");
                onlyMutate = false;
                Colony totalColony = new Colony();
                int bestId = 0;
                for (Colony colony : results) {
                    if (epoch > 20 && colony.getIndividuals().size() < bestsHistoricalCounter / 4) {
                        Map<Path, int[]> actualBest = bestsHistorical.get(bestId).entrySet().stream()
                                .sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                                .limit(bestsHistoricalCounter)
                                .collect(Collectors.toMap(a -> a.getKey(), a -> a.getValue()));
                        for (Path pathCandidate : colony.getIndividuals().keySet()) {
                            countingToTabu.merge(pathCandidate, 1, Integer::sum);
                        }
                        colony.getIndividuals().putAll(actualBest);
                    }
                    totalColony = new Colony(totalColony, colony);
                }
                List<Colony> oneBigColony = Collections.singletonList(totalColony);
                copyPaths(totalPaths, numberOfCities, paths, pathTotals, pathsCopy, pathTotalsCopy);
                createNextGeneration(gpuThreads, pathsPerThread, totalPaths, numberOfCities, paths, randomGenerator, oneBigColony, 40);
            } else {
                // Additional processing if not merging colonies
                int bestId = 0;
                Map<Path, int[]> bestPathsMap = new HashMap<>();
                for (Colony colony : results) {
                    bestsHistorical.get(bestId).putAll(colony.getIndividuals());
                    Map<Path, int[]> actualBest = bestsHistorical.get(bestId).entrySet().stream()
                            .sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                            .limit(bestsHistoricalCounter)
                            .collect(Collectors.toMap(a -> a.getKey(), a -> a.getValue()));
                    bestsHistorical.get(bestId).clear();
                    bestsHistorical.get(bestId).putAll(actualBest);
                    bestId++;
                    bestPathsMap.putAll(actualBest.entrySet().stream().sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                            .limit(1).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));
                    if (epoch > 5 && colony.getIndividuals().size() < bestsHistoricalCounter / 4) {
                        for (Path pathCandidate : colony.getIndividuals().keySet()) {
                            countingToTabu.merge(pathCandidate, 1, Integer::sum);
                        }
                        colony.getIndividuals().putAll(actualBest);
                    }
                }

                Map<Path, int[]> bestPathEntry = bestPathsMap.entrySet().stream().sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                        .limit(1).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
                Path bestPath = bestPathEntry.keySet().iterator().next();
                System.out.println("Best path = " + bestPath);
                returnResult = bestPath.toString();
                if (Instant.now().isAfter(startTime.plusSeconds(secondsCalculation))) {
                    return bestPath.toString();
                }
                StringBuilder bestSolution = new StringBuilder();
                for (int city : bestPathEntry.get(bestPath)) {
                    bestSolution.append("-").append(city);
                }
                distancesData.bestSolution = bestSolution.substring(1);
                System.out.println();

                copyPaths(totalPaths, numberOfCities, paths, pathTotals, pathsCopy, pathTotalsCopy);
                if (totalUnique < gpuThreads / 4 && onlyMutate) {
                    enableSecondPhase();
                }
                createNextGeneration(gpuThreads, pathsPerThread, totalPaths, numberOfCities, paths, randomGenerator, results, 400);
                int colonyIndex = 0;
                for (Set<Path> pathSet : allPaths) {
                    pathSet.addAll(results.get(colonyIndex).getIndividuals().keySet());
                    colonyIndex++;
                }
            }
            countingToTabu = countingToTabu.entrySet().parallelStream()
                    .sorted(Comparator.comparing(a -> a.getKey().getTotal()))
                    .limit(524288)
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
        }
        return returnResult;
    }

    private void initializePaths() {
        paths = new int[totalPaths][numberOfCities];
        pathTotals = new double[totalPaths];
        pathsCopy = new int[totalPaths][numberOfCities];
        pathTotalsCopy = new double[totalPaths];
        pathsAux = new int[totalPaths][numberOfCities];
        pathTotalsAux = new double[totalPaths];
        gaResultPaths = new int[totalPaths][numberOfCities];
        gaResultTotals = new double[totalPaths];
    }

    private void initializePopulation() {
        for (int i = 0; i < numberOfCities; i++) {
            for (int j = 0; j < totalPaths; j++) {
                paths[j][i] = (i + j) % numberOfCities;
            }
        }
    }

    private void enableSecondPhase() {
        System.out.println("------> PHASE 2 start now! <------");
        onlyMutate = false;
        epochsInGPU = 3;
        gpuThreads *= 2;
        pathsPerThread = 4;
        totalPaths = gpuThreads * pathsPerThread;
        initializePaths();
    }

    private static boolean shouldMergeColonies(int totalPaths, int mergeCounter, int totalUnique, boolean mergeColonyByTime, List<Double> cutoffsByTime, double mergeTimeRatio, int mergeStep) {
        System.out.println("Time to merge: " + mergeTimeRatio);
        boolean defaultCondition = (totalUnique < totalPaths / 32 && mergeCounter > 32) || totalUnique < totalPaths / 64;
        if (mergeColonyByTime) {
            if (cutoffsByTime.size() > mergeStep && cutoffsByTime.get(mergeStep) < mergeTimeRatio) {
                System.out.println("Step: " + mergeStep + ", time to merge: " + mergeTimeRatio + " > " + cutoffsByTime.get(mergeStep));
                return true;
            } else {
                return false;
            }
        }
        return defaultCondition;
    }

    private void checkAndRepairIntegrity(int totalPaths, int numberOfCities, int[][] paths, double[] pathTotals, Random randomGenerator, int[] integrityFaults, int colonyMultiplier, int pathsPerThread) {
        int faultCount = Arrays.stream(integrityFaults).sum();
        if (faultCount > 0) {
            System.out.println("----> ERROR CHECKING CORRECTION, integrity fault detected in " + faultCount);
            for (int i = 0; i < integrityFaults.length; i++) {
                if (integrityFaults[i] > 0) {
                    int randomIndex;
                    int colonyPart = i / (totalPaths / colonyMultiplier);
                    int lowerBound = colonyPart * (totalPaths / colonyMultiplier);
                    int upperBound = (colonyPart + 1) * (totalPaths / colonyMultiplier);
                    do {
                        randomIndex = randomGenerator.nextInt(upperBound - lowerBound) + lowerBound;
                    } while (integrityFaults[randomIndex] > 0);
                    System.arraycopy(paths[randomIndex], 0, paths[i], 0, numberOfCities);
                    pathTotals[i] = pathTotals[randomIndex];
                }
            }
        }
    }

    private double[] createBst(List<Double> list, int depth, int size) {
        double[] bstTable = new double[size];
        int base = size;
        int start = 0;
        for (int level = 0; level < depth; level++) {
            int elementsInLevel = 1 << level;
            int middle = base / 2;
            base = base / 2;
            start += elementsInLevel / 2;
            for (int j = start; j < start + elementsInLevel; j++) {
                bstTable[j] = list.get(middle);
                middle += 1 << (depth - level);
            }
        }
        return bstTable;
    }

    private static List<Colony> postEpochProcessing(int totalPaths, int[][] paths, double[] pathTotals, int epoch, int colonyMultiplier) {
        List<Map<Path, int[]>> distinctPaths = getDistinctPaths(totalPaths, pathTotals, paths, epoch, colonyMultiplier);
        List<Colony> colonies = new ArrayList<>(distinctPaths.size());
        for (Map<Path, int[]> map : distinctPaths) {
            List<Path> sequence = map.keySet().stream().sorted(Comparator.comparing(Path::getTotal)).collect(Collectors.toList());
            double best = sequence.get(0).getTotal();
            double worst = sequence.get(sequence.size() - 1).getTotal();
            colonies.add(new Colony(map, best, worst));
        }
        return colonies;
    }

    private static List<Map<Path, int[]>> getDistinctPaths(int totalPaths, double[] pathTotals, int[][] paths, int epoch, int colonyMultiplier) {
        List<Map<Path, int[]>> result = new ArrayList<>();
        Map<Double, Map<Path, int[]>> sortedMap = new HashMap<>();
        Map<Double, Path> bestPaths = new HashMap<>();
        Map<Double, Path> worstPaths = new HashMap<>();
        int pathsPerColony = totalPaths / colonyMultiplier;
        Map<Double, String> outputStrings = new HashMap<>();
        for (int col = 0; col < colonyMultiplier; col++) {
            int start = col * pathsPerColony;
            int end = (col + 1) * pathsPerColony;
            Map<Path, int[]> distinct = new HashMap<>();
            double mean = 0.0;
            for (int i = start; i < end; i++) {
                mean += pathTotals[i] / totalPaths * colonyMultiplier;
                distinct.put(new Path(pathTotals[i]), paths[i]);
            }
            result.add(distinct);
            List<Path> sequence = distinct.keySet().stream().sorted(Comparator.comparing(Path::getTotal)).collect(Collectors.toList());
            double best = sequence.get(0).getTotal();
            double worst = sequence.get(sequence.size() - 1).getTotal();
            Path bestPath = sequence.get(0);
            Path worstPath = sequence.get(sequence.size() - 1);
            String output = String.format("Colony = %3d, Epoch = %3d, Mean = %.3f, Best = %.6f, Worst = %.6f, Unique = %d", col, epoch, mean, best, worst, distinct.size());
            outputStrings.put(mean, output);
            sortedMap.put(mean, distinct);
            bestPaths.put(mean, bestPath);
            worstPaths.put(mean, worstPath);
        }
        Map<Path, int[]> previous = null;
        Path previousWorstPath = null;
        for (Double mean : outputStrings.keySet().stream().sorted().collect(Collectors.toList())) {
            System.out.println(outputStrings.get(mean));
            if (previousWorstPath != null && previous != null) {
                Path actualBest = bestPaths.get(mean);
            }
            previousWorstPath = worstPaths.get(mean);
            previous = sortedMap.get(mean);
        }
        return result;
    }

    private static void createNextGeneration(int gpuThreads, int pathsPerThread, int totalPaths, int numberOfCities, int[][] paths, Random randomGenerator, List<Colony> colonies, int scalePower) throws InterruptedException {
        int colonyIndex = 0;
        int coloniesCount = colonies.size();
        int threadsPerColony = gpuThreads / coloniesCount;
        for (Colony colony : colonies) {
            int start = colonyIndex * threadsPerColony;
            int end = (colonyIndex + 1) * threadsPerColony;
            colonyIndex++;
            List<Path> sequence = colony.getIndividuals().keySet().stream().sorted(Comparator.comparing(Path::getTotal)).collect(Collectors.toList());
            double best = sequence.get(0).getTotal();
            double worst = sequence.get(sequence.size() - 1).getTotal();
            List<Integer> threadIndices = new ArrayList<>();
            for (int j = start; j < end; j++) {
                threadIndices.add(j);
            }

            double power = Math.log(((sequence.size() / (double) totalPaths + 1.0) * (worst / best) - 1) * scalePower + 1) + 1;

            ForkJoinPool customThreadPool = new ForkJoinPool(24);
            try {
                customThreadPool.submit(() -> threadIndices.parallelStream().forEach((j) -> {
                    List<Integer> selectorList = new ArrayList<>(pathsPerThread);
                    int seqSize = sequence.size();
                    for (int k = 0; k < pathsPerThread; k++) {
                        double randomValue = randomGenerator.nextDouble();
                        int selector = (int) (Math.pow(randomValue, power) * (seqSize - 2));
                        selectorList.add(selector);
                    }
                    selectorList = selectorList.stream().sorted().distinct().collect(Collectors.toList());
                    if (seqSize <= pathsPerThread + 1) {
                        while (selectorList.size() < pathsPerThread) {
                            selectorList.add(0);
                        }
                    }
                    while (selectorList.size() < pathsPerThread) {
                        double randomValue = randomGenerator.nextDouble();
                        int selector = (int) (Math.pow(randomValue, power) * (seqSize - 2));
                        selectorList.add(selector);
                        selectorList = selectorList.stream().sorted().distinct().collect(Collectors.toList());
                    }
                    for (int k = 0; k < pathsPerThread; k++) {
                        int[] onePath = colony.getIndividuals().get(sequence.get(selectorList.get(k)));
                        System.arraycopy(onePath, 0, paths[pathsPerThread * j + k], 0, numberOfCities);
                    }
                })).get();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        }
    }

    private static void copyPaths(int totalPaths, int numberOfCities, int[][] sourcePaths, double[] sourceTotals, int[][] targetPaths, double[] targetTotals) {
        for (int i = 0; i < numberOfCities; i++) {
            for (int j = 0; j < totalPaths; j++) {
                targetPaths[j][i] = sourcePaths[j][i];
            }
        }
        System.arraycopy(sourceTotals, 0, targetTotals, 0, totalPaths);
    }
}
