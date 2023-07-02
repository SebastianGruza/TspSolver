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
    int trialsCrossover;
    int trialsMutation;
    int trialMutCounter;
    double bstTable[];
    int depthBst;
    final int[] isFault;
    final int states[];
    int existedVertices[][];
    final static int SEED_SIZE = 5;
    final static int INTEGER_SIZE = 4;


    public TspGAKernel(double[] pathSum, int[][] path, double[] gaResultSum,
                       int[][] gaResult, double[][] distances, int size, int n, int pm,
                       int epochs, int[] isFault, int trialsCrossover, int trialsMutation,
                       int mainEpoch, double bstTable[], int depthBst) {
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
        this.trialsCrossover = trialsCrossover;
        this.trialsMutation = trialsMutation;
        this.bstTable = bstTable;
        this.depthBst = depthBst;
        int maxThreads = size;
        trialMutCounter = (int) (n * sqrt(n) / 40.0 * (Math.random() * 1.6 + 0.2)) + 2;

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
            cross(gid, trialsCrossover);
            mut8(gid, 2);
            twoOpt(gid, 12);
            mut7(gid, 6);
            mut5(gid, 8);
            randomShift(gid);
            cross(gid, trialsCrossover);
            mut4(gid, 10);
            mut6(gid, 3);
            threeOpt(gid, 4);
            twoOpt(gid, 12);


            if (epoch + 1 == epochs) {
                checkIntegrity(gid);
            }
        }
    }

    private void mut8(int gid, int trials) {
        for (int trial = 0; trial < trials; trial++) {
            calculateMutation8(gid);
            calculatePathDistances(gid);
            selectionOfIndividuals(gid);
            randomShift(gid);
        }
    }

    private void mut7(int gid, int trials) {
        for (int trial = 0; trial < trials; trial++) {
            calculateMutation7(gid);
            calculatePathDistances(gid);
            selectionOfIndividuals(gid);
            randomShift(gid);
        }
    }

    private void mut5(int gid, int trials) {
        for (int trial = 0; trial < trials; trial++) {
            calculateMutation5(gid);
            calculatePathDistances(gid);
            selectionOfIndividuals(gid);
            randomShift(gid);
        }
    }

    private void mut4(int gid, int trials) {
        for (int trial = 0; trial < trials; trial++) {
            calculateMutation4(gid);
            calculatePathDistances(gid);
            selectionOfIndividuals(gid);
            randomShift(gid);
        }
    }

    private void mut6(int gid, int trials) {
        for (int trial = 0; trial < trials; trial++) {
            calculateMutation6(gid);
            calculatePathDistances(gid);
            selectionOfIndividuals(gid);
            randomShift(gid);
        }
    }

    private void cross(int gid, int trials) {
        //step 3: crossover:
        for (int trial = 0; trial < trials; trial++) {
            if (pm == 4) {
                calculateCrossoverFor4(gid, trial);
            } else if (pm == 2) {
                crossover(gid, 0, 1);
            } else {
                //TODO:
            }
            calculatePathDistances(gid);
            selectionOfIndividuals(gid);
            randomShift(gid);
        }
    }

    private void threeOpt(int gid, int trials) {
        for (int trial = 0; trial < trials; trial++) {
            calculate3OptMutation(gid);
            calculatePathDistances(gid);
            selectionOfIndividuals(gid);
            randomShift(gid);
        }
    }

    private void twoOpt(int gid, int trials) {
        for (int trial = 0; trial < trials; trial++) {
            calculateMutation3(gid);
            calculatePathDistances(gid);
            selectionOfIndividuals(gid);
            randomShift(gid);
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
            int rnd2 = getBestRandomVertex(gaResult[parent], rnd1, n / 50);

            int zamien = gaResult[parent][rnd1];
            gaResult[parent][rnd1] = gaResult[parent][rnd2];
            gaResult[parent][rnd2] = zamien;

        }
    }

    private void calculate3OptMutation(int gid) {
        for (int el = 0; el < pm; el++) {
            int parent = pm * gid + el;
            for (int i = 0; i < n; i++) {
                existedVertices[parent][i] = 0;
            }
            double best = Double.MAX_VALUE;
            int rnd1 = 0;
            int rnd2 = rnd1;
            int rnd3 = rnd2;

            for (int i = 0; i < trialMutCounter * 4; i++) {
                float numRandom1 = random01();
                float numRandom2 = random01();
                float numRandom3 = random01();

                int actualRnd1 = (int) (numRandom1 * (n - 6)) + 1;
                int actualRnd2 = (int) (numRandom2 * (n - actualRnd1 - 3)) + actualRnd1 + 1;
                int actualRnd3 = (int) (numRandom3 * (n - actualRnd2 - 2)) + actualRnd2 + 1;

                double actualBest = getBestRandomVertexThreeOpt(gaResult[parent], actualRnd1, actualRnd2, actualRnd3);
                if (actualBest < best) {
                    best = actualBest;
                    rnd1 = actualRnd1;
                    rnd2 = actualRnd2;
                    rnd3 = actualRnd3;
                }
            }

            perform3OptSwap(parent, rnd1, rnd2, rnd3);

        }
    }

    private double getBestRandomVertexThreeOpt(int[] ints, int i, int j, int k) {
        int a = ints[i];
        int b = ints[Next(i)];
        int c = ints[j];
        int d = ints[Next(j)];
        int e = ints[k];
        int f = ints[Next(k)];

        double existingDistance = distances[a][b] + distances[c][d] + distances[e][f];
        double newDistance1 = distances[a][c] + distances[b][e] + distances[d][f];

        return newDistance1 - existingDistance;
    }

    private void perform3OptSwap(int parent, int i, int j, int k) {
        // swapping i+1 to j
        int m = j;
        int n = i+1;
        while (m >= n) {
            int temp = gaResult[parent][n];
            gaResult[parent][n] = gaResult[parent][m];
            gaResult[parent][m] = temp;
            m--;
            n++;
        }

        // swapping j+1 to k
        m = k;
        n = j+1;
        while (m >= n) {
            int temp = gaResult[parent][n];
            gaResult[parent][n] = gaResult[parent][m];
            gaResult[parent][m] = temp;
            m--;
            n++;
        }
    }

    private void calculateMutation6(int gid) {
        for (int el = 0; el < pm; el++) {
            int parent = pm * gid + el;
            double best = Double.MAX_VALUE;
            int rnd1 = 0;
            int rnd2 = rnd1;
            for (int i = 0; i < trialMutCounter / 2; i++) {
                float numRandom1 = random01();
                int actualRnd1 = (int) (numRandom1 * (n / 2));
                float numRandom2 = random01();
                int actualRnd2 = (int) (numRandom2 * (n - 2 - actualRnd1)) + actualRnd1;
                double actualBest = getBestRandomVertexBetterMethod(gaResult[parent], actualRnd1, actualRnd2);
                if (actualBest < best) {
                    best = actualBest;
                    rnd1 = actualRnd1;
                    rnd2 = actualRnd2;
                }
            }
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
            int p2 = getBestRandomVertex(gaResult[parent], p1, n / 50);
            if (p2 != p1 && p2 != q1) {
                int temp = gaResult[parent][q1];
                gaResult[parent][q1] = gaResult[parent][p2];
                int check = 1;
                for (int i = 0; i < n; i++) {
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
                existedVertices[parent][i] = 0;
            }
            double best = Double.MAX_VALUE;
            int rnd1 = 0;
            int rnd2 = rnd1;
            for (int i = 0; i < trialMutCounter; i++) {
                float numRandom1 = random01();
                float numRandom2 = random01();
                int actualRnd1 = (int) (numRandom1 * (n - 3)) + 1;
                int actualRnd2 = (int) (numRandom2 * (n - actualRnd1 - 2)) + actualRnd1 + 1;
                double actualBest = getBestRandomVertexTwoOpt(gaResult[parent], actualRnd1, actualRnd2);
                if (actualBest < best) {
                    best = actualBest;
                    rnd1 = actualRnd1;
                    rnd2 = actualRnd2;
                }
            }
            if (rnd1 != rnd2) {
                int check = 0;
                for (int i = 0; i < n; i++) {
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

    private void calculateMutation4(int gid) {
        for (int el = 0; el < pm; el++) {
            int parent = pm * gid + el;
            double best = Double.MAX_VALUE;
            int p1 = 0;
            int p2 = p1;
            for (int i = 0; i < trialMutCounter / 2; i++) {
                float numRandom1 = random01();
                float numRandom2 = random01();
                int actualP1 = (int) (numRandom1 * (n / 2));
                int actualP2 = (int) (numRandom2 * (n - actualP1 - 2)) + actualP1;
                double actualBest = getBestRandomVertexGetMiddleCity(gaResult[parent], actualP1, actualP2);
                if (actualBest < best) {
                    best = actualBest;
                    p1 = actualP1;
                    p2 = actualP2;
                }
            }
            int q1 = Next(p1);
            int r1 = Next(q1);
            int r2 = Next(p2);
            int p2City = gaResult[parent][p2];
            int r2City = gaResult[parent][r2];
            if (p2 != p1 && p2 != q1) {
                int temp = gaResult[parent][q1];
                gaResult[parent][q1] = gaResult[parent][r1];
                int check = 1;
                for (int i = 0; i < n; i++) {
                    if (gaResult[parent][Next(r1)] != p2City && check == 1) {
                        gaResult[parent][r1] = gaResult[parent][Next(r1)];
                        r1 = Next(r1);
                    } else {
                        check = 0;
                    }
                }
                gaResult[parent][r1] = p2City;
                gaResult[parent][Next(r1)] = temp;
                gaResult[parent][Next(Next(r1))] = r2City;
            }
        }
    }

    private void calculateMutation7(int gid) {
        for (int el = 0; el < pm; el++) {
            int parent = pm * gid + el;

            double best = Double.MAX_VALUE;
            int p1 = 0;
            int p2 = p1;
            for (int i = 0; i < trialMutCounter / 2; i++) {
                float numRandom1 = random01();
                float numRandom2 = random01();
                int actualP1 = (int) (numRandom1 * (n / 2));
                int actualP2 = (int) (numRandom2 * (n - actualP1 - 7)) + actualP1 + 4;
                double actualBest = getBestRandomVertexGet1to3City(gaResult[parent], actualP1, actualP2);
                if (actualBest < best) {
                    best = actualBest;
                    p1 = actualP1;
                    p2 = actualP2;
                }
            }
            int q1 = Next(p1);
            int r1 = Next(q1);
            int s1 = Next(r1);
            int r2 = Next(p2);
            int p2City = gaResult[parent][p2];
            int r2City = gaResult[parent][r2];
            if (p2 != p1 && p2 != q1) {
                int temp1 = gaResult[parent][q1];
                int temp2 = gaResult[parent][r1];
                int temp3 = gaResult[parent][s1];
                gaResult[parent][q1] = gaResult[parent][Next(s1)];
                gaResult[parent][r1] = gaResult[parent][Next(Next(s1))];
                int check = 1;
                for (int i = 0; i < n; i++) {
                    if (gaResult[parent][Next(Next(Next(s1)))] != p2City && check == 1) {
                        gaResult[parent][s1] = gaResult[parent][Next(Next(Next(s1)))];
                        s1 = Next(s1);
                    } else {
                        check = 0;
                    }
                }
                gaResult[parent][s1] = p2City;
                gaResult[parent][Next(s1)] = temp1;
                gaResult[parent][Next(Next(s1))] = temp2;
                gaResult[parent][Next(Next(Next(s1)))] = temp3;
                gaResult[parent][Next(Next(Next(Next((s1)))))] = r2City;
            }
        }
    }

    private void calculateMutation8(int gid) {
        for (int el = 0; el < pm; el++) {
            int parent = pm * gid + el;
            for (int i = 0; i < n; i++) {
                existedVertices[parent][i] = 0;
            }
            double best = Double.MAX_VALUE;
            int p1 = 0;
            int p2 = p1;
            int p3 = 2;
            for (int i = 0; i < trialMutCounter * 4; i++) {
                float numRandom1 = random01();
                float numRandom2 = random01();
                float numRandom3 = random01();
                int actualP1 = (int) (numRandom1 * (n / 3));
                int actualSize = (int) (numRandom2 * numRandom2 * numRandom2 * (n / 40) + 0.6) + 2;
                int actualP2 = (int) (numRandom3 * (n - actualP1 - actualSize - 7)) + actualP1 + actualSize + 2;
                double actualBest = getBestRandomVertexGet0toNCity(gaResult[parent], actualP1, actualP2, actualSize);
                if (actualBest < best) {
                    best = actualBest;
                    p1 = actualP1;
                    p2 = actualP2;
                    p3 = actualSize;
                }
            }
            int q1 = Next(p1);
            int r2 = Next(p2);
            int p2City = gaResult[parent][p2];
            int r2City = gaResult[parent][r2];
            if (p2 != p1 && p2 != q1) {

                for (int i = 0; i <= p3; i++) {
                    existedVertices[parent][(q1 + i) % n] = gaResult[parent][(q1 + i) % n];
                }

                int s1 = (q1 + p3 + 1) % n;
                for (int i = 0; i <= p3; i++) {
                    gaResult[parent][(q1 + i) % n] = gaResult[parent][(s1 + i) % n];
                }

                int check = 1;
                for (int i = 0; i < n; i++) {
                    if (gaResult[parent][(s1 + p3 + 1) % n] != p2City && check == 1) {
                        gaResult[parent][s1] = gaResult[parent][(s1 + p3 + 1) % n];
                        s1 = Next(s1);
                    } else {
                        check = 0;
                    }
                }
                gaResult[parent][s1] = p2City;

                for (int i = 0; i <= p3; i++) {
                    gaResult[parent][(s1 + i + 1) % n] = existedVertices[parent][q1 + i];
                }

                gaResult[parent][(s1 + p3 + 2) % n] = r2City;
            }
        }
    }


    private void calculateMutation5(int gid) {
        for (int el = 0; el < pm; el++) {
            int parent = pm * gid + el;

            double best = Double.MAX_VALUE;
            int p1 = 0;
            int p2 = p1;
            for (int i = 0; i < trialMutCounter / 2; i++) {
                float numRandom1 = random01();
                float numRandom2 = random01();
                int actualP1 = (int) (numRandom1 * (n / 2));
                int actualP2 = (int) (numRandom2 * (n - actualP1 - 6)) + actualP1 + 3;
                double actualBest = getBestRandomVertexGetMiddle2City(gaResult[parent], actualP1, actualP2);
                if (actualBest < best) {
                    best = actualBest;
                    p1 = actualP1;
                    p2 = actualP2;
                }
            }
            int q1 = Next(p1);
            int r1 = Next(q1);
            int r2 = Next(p2);
            int p2City = gaResult[parent][p2];
            int r2City = gaResult[parent][r2];
            if (p2 != p1 && p2 != q1) {
                int temp1 = gaResult[parent][q1];
                int temp2 = gaResult[parent][r1];
                gaResult[parent][q1] = gaResult[parent][Next(r1)];
                int check = 1;
                for (int i = 0; i < n; i++) {
                    if (gaResult[parent][Next(Next(r1))] != p2City && check == 1) {
                        gaResult[parent][r1] = gaResult[parent][Next(Next(r1))];
                        r1 = Next(r1);
                    } else {
                        check = 0;
                    }
                }
                gaResult[parent][r1] = p2City;
                gaResult[parent][Next(r1)] = temp1;
                gaResult[parent][Next(Next(r1))] = temp2;
                gaResult[parent][Next(Next(Next(r1)))] = r2City;
            }
        }
    }

    private int getBestRandomVertex(int[] ints, int rnd1, int counter) {
        int vertex1 = ints[rnd1];
        int rnd2 = Next(rnd1);
        int bestRnd2 = rnd2;
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

    private double getBestRandomVertexBetterMethod(int[] ints, int rnd1, int rnd21) {
        int vertex11 = ints[rnd1];
        int rnd12 = Next(rnd1);
        int vertex12 = ints[rnd12];
        int rnd10 = Previous(rnd1);
        int vertex10 = ints[rnd10];
        int vertex21 = ints[rnd21];
        int rnd22 = Next(rnd21);
        int vertex22 = ints[rnd22];
        int rnd20 = Previous(rnd21);
        int vertex20 = ints[rnd20];
        double actual1 = distances[vertex10][vertex11] + distances[vertex11][vertex12];
        double actual2 = distances[vertex20][vertex21] + distances[vertex21][vertex22];
        double new1 = distances[vertex20][vertex11] + distances[vertex11][vertex22];
        double new2 = distances[vertex10][vertex21] + distances[vertex21][vertex12];
        double actualBest = new1 + new2 - actual1 - actual2;
        return actualBest;
    }

    private double getBestRandomVertexTwoOpt(int[] ints, int rnd1, int rnd21) {
        int vertex11 = ints[rnd1];
        int rnd12 = Previous(rnd1);
        int vertex12 = ints[rnd12];
        double actual1 = distances[vertex11][vertex12];
        int vertex21 = ints[rnd21];
        int rnd22 = Next(rnd21);
        int vertex22 = ints[rnd22];
        double actual2 = distances[vertex21][vertex22];
        double new1 = distances[vertex11][vertex22];
        double new2 = distances[vertex12][vertex21];
        double actualBest = new1 + new2 - actual1 - actual2;
        return actualBest;
    }

    private double getBestRandomVertexGetMiddleCity(int[] ints, int rnd1, int rnd21) {
        int vertex11 = ints[rnd1];
        int rnd12 = Next(rnd1);
        int vertex12 = ints[rnd12];
        int rnd13 = Next(rnd12);
        int vertex13 = ints[rnd13];
        double actual1 = distances[vertex11][vertex12] + distances[vertex12][vertex13];
        double newPart1 = distances[vertex11][vertex13];
        int vertex21 = ints[rnd21];
        int rnd22 = Next(rnd21);
        int vertex22 = ints[rnd22];
        double actual2 = distances[vertex21][vertex22];
        double newPart2 = distances[vertex21][vertex12] + distances[vertex12][vertex22];
        double actualBest = newPart1 + newPart2 - actual1 - actual2;
        return actualBest;
    }

    private double getBestRandomVertexGet1to3City(int[] ints, int rnd1, int rnd21) {
        int vertex11 = ints[rnd1];
        int rnd12 = Next(rnd1);
        int vertex12 = ints[rnd12];
        int rnd13 = Next(rnd12);
        int vertex13 = ints[rnd13];
        int rnd14 = Next(rnd13);
        int vertex14 = ints[rnd14];
        int rnd15 = Next(rnd14);
        int vertex15 = ints[rnd15];
        double actual1 = distances[vertex11][vertex12] + distances[vertex12][vertex13] + distances[vertex13][vertex14] + distances[vertex14][vertex15];
        double newPart1 = distances[vertex11][vertex15];
        int vertex21 = ints[rnd21];
        int rnd22 = Next(rnd21);
        int vertex22 = ints[rnd22];
        double actual2 = distances[vertex21][vertex22];
        double newPart2 = distances[vertex21][vertex12] + distances[vertex12][vertex13] + distances[vertex13][vertex14] + distances[vertex14][vertex22];
        double actualBest = newPart1 + newPart2 - actual1 - actual2;
        return actualBest;
    }

    private double getBestRandomVertexGet0toNCity(int[] ints, int rnd11, int rnd21, int size) {
        int vertex11 = ints[rnd11];
        int rnd12 = Next(rnd11);
        int vertex12 = ints[rnd12];

        int rnd14 = rnd12 + size;
        int vertex14 = ints[rnd14];
        int rnd15 = Next(rnd14);
        int vertex15 = ints[rnd15];
        double actual1 = distances[vertex11][vertex12] + distances[vertex14][vertex15];
        double newPart1 = distances[vertex11][vertex15];
        int vertex21 = ints[rnd21];
        int rnd22 = Next(rnd21);
        int vertex22 = ints[rnd22];
        double actual2 = distances[vertex21][vertex22];
        double newPart2 = distances[vertex21][vertex12] + distances[vertex14][vertex22];
        double actualBest = newPart1 + newPart2 - actual1 - actual2;
        return actualBest;
    }

    private double getBestRandomVertexGetMiddle2City(int[] ints, int rnd1, int rnd21) {
        int vertex11 = ints[rnd1];
        int rnd12 = Next(rnd1);
        int vertex12 = ints[rnd12];
        int rnd13 = Next(rnd12);
        int vertex13 = ints[rnd13];
        int rnd14 = Next(rnd13);
        int vertex14 = ints[rnd14];
        double actual1 = distances[vertex11][vertex12] + distances[vertex12][vertex13] + distances[vertex13][vertex14];
        double newPart1 = distances[vertex11][vertex14];
        int vertex21 = ints[rnd21];
        int rnd22 = Next(rnd21);
        int vertex22 = ints[rnd22];
        double actual2 = distances[vertex21][vertex22];
        double newPart2 = distances[vertex21][vertex12] + distances[vertex12][vertex13] + distances[vertex13][vertex22];
        double actualBest = newPart1 + newPart2 - actual1 - actual2;
        return actualBest;
    }

    private int Next(int actual) {
        return (actual == this.n - 1 ? 0 : actual + 1);
    }

    private int Previous(int actual) {
        return (actual == 0 ? this.n - 1 : actual - 1);
    }


    private void calculateCrossoverFor4(int gid, int trial) {
        if (trial % 3 == 0) {
            crossover(gid, 2, 3);
            crossover(gid, 0, 1);
        } else if (trial % 3 == 1) {
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
            //TABU PATH:
            if (searchInBst(pathSum[noPath]) == 1) {
                pathSum[noPath] *= 1.004;
            }
            if (searchInBst(gaResultSum[noPath]) == 1) {
                gaResultSum[noPath] *= 1.004;
            }
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

    private int searchInBst(double value) {
        int startId = 0;
        for (int level = 0; level < depthBst; level++) {
            double vertexInBst = bstTable[startId];
            int leftId = (startId + 1) * 2 - 1;
            int rightId = (startId + 1) * 2;
            if (vertexInBst <= value + 0.0000001 &&
                    vertexInBst >= value - 0.0000001) {
                return 1;
            } else if (vertexInBst > value && leftId < 1 << depthBst) {
                startId = leftId;
            } else if (vertexInBst < value && rightId < 1 << depthBst) {
                startId = rightId;
            } else {
                return 0;
            }
        }
        return 0;
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
