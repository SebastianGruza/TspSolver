# TspSolver

TspSolver is a cutting-edge project leveraging the power of GPU acceleration for solving the Traveling Salesman Problem (TSP) using Genetic Algorithms (GA) and APARAPI. The project utilizes Java and the Spring Framework.

## Features
- Reads .tsp files from the current directory.
- Saves optimization results to "results.txt".
- Visualizes real-time computations and the shortest path found so far at `http://127.0.0.1:8080`.
- Innovative improvements to GA, including:
  - Division of the population into colonies for independent calculation.
  - Under certain rules, colonies can merge.
  - Tabu paths to penalize certain paths for a prolonged lack of improvement, aiding escape from local minima.

## Configuration
The project uses an `application.properties` configuration file. Here are some of the configurations:

```properties
tsp.filename=full.txd    # The problem file name
tsp.gpuThreads=512       # The number of GPU workers to be used
tsp.colonyMultiplier=4  # The number of colonies to be used in the problem
tsp.divideGreedy=10     # The divisor of the greedy algorithm by which the number of points of a given problem will be divided
tsp.scaleTime=0.01      # Time restriction scale
tsp.mergeColonyByTime=true  # Whether to turn on the function of merging colonies according to time
tsp.cutoffsByTime=0.4,0.65,0.82,0.95  # The time points when colonies will be merged
```

### Installation

1. Clone the repository: `git clone https://github.com/SebastianGruza/TspSolver.git`
2. Install the required dependencies
3. Build the project in maven `mvn clean install`
4. Configure the `application.properties` file as needed
5. Run the application (provide instructions on how to run the project)

### Usage

1. Place the TSP problem file in the project folder
2. Configure the desired parameters in `application.properties`
3. Run the application to start the TSP solving process
4. Access `http://127.0.0.1:8080` to visualize the problem and the current best-known solution

### Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow the guidelines outlined in `CONTRIBUTING.md`.

### License

This project is licensed under the [MIT License](LICENSE).
