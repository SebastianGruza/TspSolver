## TspSolver

TspSolver is a Java and Spring-based project that solves the Traveling Salesman Problem (TSP) using a Genetic Algorithm (GA) accelerated by APARAPI and GPU. It leverages APARAPI for GPU acceleration.

### Description

TspSolver is designed to solve TSP instances by optimizing the path of the traveling salesman. It reads TSP files (.tsp) from the current folder and saves the optimization results to `results.txt`. Additionally, it provides a web-based visualization of the problem being solved, along with the shortest path found so far, accessible at `http://127.0.0.1:8080`.

### Features

- Utilizes a Genetic Algorithm (GA) with GPU acceleration through APARAPI
- Reads TSP files and saves optimization results
- Web-based visualization of the problem and the current best-known solution
- Improvements over typical GAs, including:
  - Population division into colonies computed independently
  - Colony merging based on specific rules
  - Tabu paths to escape local minima
- Configurable parameters in the `application.properties` file:
  - `tsp.filename`: Name of the TSP problem file
  - `tsp.gpuThreads`: Number of GPU worker threads
  - `tsp.colonyMultiplier`: Number of colonies used in the problem
  - `tsp.divideGreedy`: Divisor for the greedy algorithm used for point division
  - `tsp.scaleTime`: Time constraint scaling factor
  - `tsp.mergeColonyByTime`: Enable colony merging based on time
  - `tsp.cutoffsByTime`: Time points for colony merging (e.g., 0.4, 0.65, 0.82, 0.95)


### Installation

1. Clone the repository: `git clone https://github.com/SebastianGruza/TspSolver.git`
2. Install the required dependencies (provide instructions if necessary)
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
