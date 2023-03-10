package com.tsp.solver;

import com.aparapi.Kernel;
import org.uncommons.maths.binary.BinaryUtils;
import org.uncommons.maths.random.DefaultSeedGenerator;

class TspGAKernel extends Kernel {

    double[][] distances;
    int[][] path;
    double[] pathSum;
    int[][] gaResult;
    double[] gaResultSum;
    int size;
    int n;
    int pm;
    int epochs;
    final int[] isFault;
    final int states[];
    int existedVertices[][];
    final static int SEED_SIZE = 5;
    final static int INTEGER_SIZE = 4;


    public TspGAKernel(double[] pathSum, int[][] path, double[] gaResultSum, int[][] gaResult, double[][] distances, int size, int n, int pm, int epochs, int[] isFault) {
        this.pathSum = pathSum;
        this.distances = distances;
        this.path = path;
        this.size = size;
        this.n = n;
        this.pm = pm;
        this.gaResultSum = gaResultSum;
        this.gaResult = gaResult;
        this.epochs = epochs;
        this.existedVertices = new int[size * pm][n];
        this.isFault = isFault;
        int maxThreads = size;

        states = new int[SEED_SIZE * maxThreads];

        int[] seeds = BinaryUtils.convertBytesToInts(DefaultSeedGenerator.getInstance().generateSeed(SEED_SIZE * maxThreads * INTEGER_SIZE));

        if (SEED_SIZE * maxThreads != seeds.length)
            throw new IllegalArgumentException(String.format("Wrong size of seeds for threads! Expected %d, got %d, for %d threads.", SEED_SIZE * maxThreads, seeds.length, maxThreads));

        for (int number = 0; number < states.length; number++)
            states[number] = seeds[number];
    }

    @Override
    public void run() {
        int gid = getGlobalId();
        for (int epoch = 0; epoch < epochs; epoch++) {
            randomShift(gid);
            //step 2: mutation:
            int trialsMutation = 12;
            for (int trial = 0; trial < trialsMutation; trial++) {
                calculateMutation(gid);
                calculatePathDistances(gid);
                selectionOfIndividuals(gid);
                randomShift(gid);
            }

            for (int trial = 0; trial < trialsMutation; trial++) {
                calculateMutation2(gid);
                calculatePathDistances(gid);
                selectionOfIndividuals(gid);
                randomShift(gid);
            }

            for (int trial = 0; trial < trialsMutation; trial++) {
                calculateMutation3(gid);
                calculatePathDistances(gid);
                selectionOfIndividuals(gid);
                randomShift(gid);
            }

            //step 3: crossover:
            int trialsCrossover = 6;
            for (int trial = 0; trial < trialsCrossover; trial++) {
                if (pm == 4) {
                    calculateCrossoverFor4(gid, epoch);
                } else if (pm == 2) {
                    crossover(gid, 0, 1);
                } else {
                    //TODO:
                }
                calculatePathDistances(gid);
                selectionOfIndividuals(gid);
                randomShift(gid);
            }
            if (epoch + 1 == epochs) {
                checkIntegrity(gid);
            }
        }
    }

    private void checkIntegrity(int gid) {
        for (int el = 0; el < pm; el++) {
            int parent = pm * gid + el;
            for (int i = 0; i < n; i++) {
                existedVertices[parent][i] = 0;
            }
            for (int i = 0; i < n; i++) {
                if (existedVertices[parent][path[parent][i]] == 0) {
                    existedVertices[parent][path[parent][i]] = 1;
                } else {
                    isFault[parent] = 1;
                }
            }
        }
    }

    private void calculateMutation(int gid) {
        for (int el = 0; el < pm; el++) {
            int parent = pm * gid + el;
            float numRandom1 = random01();
            int rnd1 = (int) (numRandom1 * (n - 2));
            int rnd2 = getBestRandomVertex(gaResult[parent], rnd1, n/50);

            int zamien = gaResult[parent][rnd1];
            gaResult[parent][rnd1] = gaResult[parent][rnd2];
            gaResult[parent][rnd2] = zamien;

        }
    }

    private void calculateMutation2(int gid) {
        for (int el = 0; el < pm; el++) {
            int parent = pm * gid + el;
            float numRandom1 = random01();
            float numRandom2 = random01();
            int p1 = (int) (numRandom1 * (n - 1));
            int q1 = Next(p1);
            int p2 = getBestRandomVertex(gaResult[parent], p1, n/50);
            if (p2 != p1 && p2 != q1) {
                int temp = gaResult[parent][q1];
                gaResult[parent][q1] = gaResult[parent][p2];
                int check = 1;
                for (int i = 0; i < size; i++) {
                    if (gaResult[parent][Previous(p2)] != temp && check == 1) {
                        gaResult[parent][p2] = gaResult[parent][Previous(p2)];
                        p2 = Previous(p2);
                    } else {
                        check = 0;
                    }
                }
                gaResult[parent][p2] = temp;
            }
        }
    }

    private void calculateMutation3(int gid) {
        for (int el = 0; el < pm; el++) {
            int parent = pm * gid + el;
            for (int i = 0; i < n; i++) {
                existedVertices[parent][0] = 0;
            }
            float numRandom1 = random01();
            int rnd1 = (int) (numRandom1 * (n - 2));
            int rnd2 = getBestRandomVertex(gaResult[parent], rnd1, n/50);

            if  (rnd1 != rnd2 ) {
                int check = 0;
                for (int i = 0; i < size; i++) {
                    if (existedVertices[parent][rnd1] == 0 && existedVertices[parent][rnd2] == 0 && check == 0) {
                        if (rnd1 == rnd2 || Next(rnd1) == rnd2) {
                            check = 1;
                        }
                        int zamien = gaResult[parent][rnd1];
                        gaResult[parent][rnd1] = gaResult[parent][rnd2];
                        gaResult[parent][rnd2] = zamien;
                        existedVertices[parent][rnd1] = 1;
                        existedVertices[parent][rnd2] = 1;
                        rnd1 = Next(rnd1);
                        rnd2 = Previous(rnd2);
                    }
                }
            }
        }
    }

    private int getBestRandomVertex(int[] ints, int rnd1, int counter) {
        int vertex1 = ints[rnd1];
        int rnd2 = Next(rnd1);
        int bestRnd2  = rnd2;
        double best = Double.MAX_VALUE;
        for (int i = 0; i < counter; i++) {
            float numRandom2 = random01();
            rnd2 = (int) (numRandom2 * (n - 2));
            int vertex2 = ints[rnd2];
            double actualBest = distances[vertex1][vertex2];
            if (actualBest < best && rnd1 != rnd2) {
                best = actualBest;
                bestRnd2 = rnd2;
            }
        }
        rnd2 = bestRnd2;
        return rnd2;
    }

    private int Next(int actual) {
        return (actual == this.n - 1 ? 0 : actual + 1);
    }

    private int Previous(int actual) {
        return (actual == 0 ? this.n - 1 : actual - 1);
    }


    private void calculateCrossoverFor4(int gid, int epoch) {
        if (epoch % 3 == 0) {
            crossover(gid, 2, 3);
            crossover(gid, 0, 1);
        } else if (epoch % 3 == 1) {
            crossover(gid, 1, 3);
            crossover(gid, 0, 2);
        } else {
            crossover(gid, 0, 3);
            crossover(gid, 1, 2);
        }
    }

    private void crossover(int gid, int el1, int el2) {
        int parent1 = pm * gid + el1;
        int parent2 = pm * gid + el2;
        float numRandom = random01();
        int cutSize = (int) ((numRandom * 0.3 * n) + n / 2);
        float numRandom2 = random01();
        int cutPoint = (int) (numRandom2 * (n - cutSize - 3)) + 1;
        crossoverOne(parent1, parent2, cutPoint, cutSize);
        crossoverOne(parent2, parent1, cutPoint, cutSize);
    }

    private void crossoverOne(int parent1, int parent2, int cutPoint, int cutSize) {

        int k = cutPoint + cutSize;
        for (int i = 0; i < n; i++) {
            existedVertices[parent1][i] = 0;
            existedVertices[parent2][i] = 0;
        }
        for (int i = cutPoint + cutSize; i < n; i++) {
            existedVertices[parent2][path[parent2][i]] = 1;
        }

        for (int j = cutPoint; j < cutPoint + cutSize; j++) {
            if (existedVertices[parent2][gaResult[parent1][j]] == 1) {
                existedVertices[parent1][gaResult[parent1][j]] = 1;
            }
        }
        for (int i = cutPoint + cutSize; i < n; i++) {
            if (existedVertices[parent1][path[parent2][i]] == 0) {
                gaResult[parent1][k] = path[parent2][i];
                existedVertices[parent1][path[parent2][i]] = 1;
                k++;
                if (k == n)
                    k = 0;
            }
        }

        for (int i = 0; i < n; i++) {
            existedVertices[parent1][i] = 0;
            existedVertices[parent2][i] = 0;
        }

        for (int i = 0; i < cutPoint + cutSize; i++) {
            existedVertices[parent2][path[parent2][i]] = 1;
        }

        for (int j = cutPoint; j < cutPoint + cutSize; j++) {
            if (existedVertices[parent2][gaResult[parent1][j]] == 1) {
                existedVertices[parent1][gaResult[parent1][j]] = 1;
            }
        }

        for (int i = 0; i < cutPoint + cutSize; i++) {
            if (existedVertices[parent1][path[parent2][i]] == 0) {
                gaResult[parent1][k] = path[parent2][i];
                existedVertices[parent1][path[parent2][i]] = 1;
                k++;
                if (k == n)
                    k = 0;
            }
        }
    }

    private void randomShift(int gid) {
        //step: random shift:
        for (int el = 0; el < pm; el++) {
            float numRandom = random01();
            int intRandom = (int) (numRandom * 0.9 * n);
            float randomToInverse = random01();
            int isInverse = 0;
            if (randomToInverse < 0.5) {
                isInverse = 1;
            }
            int noPath = pm * gid + el;
            for (int vertex1 = 0; vertex1 < n; vertex1++) {
                if (isInverse == 1) {
                    int vertex2 = (n - 1) - ((vertex1 + intRandom) % n);
                    gaResult[noPath][vertex1] = path[noPath][vertex2];
                } else {
                    int vertex2 = (vertex1 + intRandom) % n;
                    gaResult[noPath][vertex1] = path[noPath][vertex2];
                }
            }
            for (int vertex = 0; vertex < n; vertex++) {
                path[noPath][vertex] = gaResult[noPath][vertex];
            }
        }
    }

    private void calculatePathDistances(int gid) {
        //step: calculate path:
        for (int el = 0; el < pm; el++) {
            int noPath = pm * gid + el;
            pathSum[noPath] = 0;
            gaResultSum[noPath] = 0;
            for (int index = 0; index < n - 1; index++) {
                pathSum[noPath] += distances[path[noPath][index]][path[noPath][index + 1]];
                gaResultSum[noPath] += distances[gaResult[noPath][index]][gaResult[noPath][index + 1]];
            }
            pathSum[noPath] += distances[path[noPath][0]][path[noPath][n - 1]];
            gaResultSum[noPath] += distances[gaResult[noPath][0]][gaResult[noPath][n - 1]];
        }
    }

    private void selectionOfIndividuals(int gid) {
        //step (only when this epoch not last): selection of individuals:
        for (int el = 0; el < pm; el++) {
            int noPath = pm * gid + el;
            int sumToSelect = 0;
            int weakest = noPath;
            for (int el2 = 0; el2 < pm; el2++) {
                int noPath2 = pm * gid + el2;
                if (el2 != el) {
                    if (pathSum[noPath2] <= gaResultSum[noPath] + 0.0000001
                            && pathSum[noPath2] >= gaResultSum[noPath] - 0.0000001) {
                        sumToSelect += 1;
                    }
                    if (pathSum[noPath2] > pathSum[noPath] && pathSum[noPath2] > pathSum[weakest]) {
                        weakest = noPath2;
                    }
                }
            }
            if (pathSum[noPath] > gaResultSum[noPath] && sumToSelect == 0) {
                for (int index = 0; index < n; index++) {
                    path[noPath][index] = gaResult[noPath][index];
                }
                pathSum[noPath] = gaResultSum[noPath];
            } else if (pathSum[weakest] > gaResultSum[noPath] && sumToSelect == 0) {
                for (int index = 0; index < n; index++) {
                    path[weakest][index] = gaResult[noPath][index];
                }
                pathSum[weakest] = gaResultSum[noPath];
            }
        }
    }

    /**
     * Implements PRNGKernel method
     */
    private int random() {
        int offset = SEED_SIZE * getGlobalId();

        int t = (states[offset + 0] ^ (states[offset + 0] >> 7));

        states[offset + 0] = states[offset + 1];
        states[offset + 1] = states[offset + 2];
        states[offset + 2] = states[offset + 3];
        states[offset + 3] = states[offset + 4];
        states[offset + 4] = (states[offset + 4] ^ (states[offset + 4] << 6)) ^ (t ^ (t << 13));

        return (states[offset + 1] + states[offset + 1] + 1) * states[offset + 4];
    }

    private float random01() {
        float value = random();
        if (value < 0)
            return value / Integer.MIN_VALUE;
        else if (value > 0)
            return value / Integer.MAX_VALUE;
        else
            return value;
    }
}
