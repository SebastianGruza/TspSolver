# TspSolver

**TspSolver** is a high-performance solver for the Traveling Salesman Problem (TSP) that implements a **Massively Parallel Hybrid Memetic Algorithm**. It leverages the [Aparapi](https://aparapi.github.io/) library to offload the entire evolutionary process to the GPU, achieving significant acceleration.

This project is not a simple Genetic Algorithm (GA). It is an advanced **Memetic Algorithm (MA)** that combines global search (genetic operators like crossover) with intensive local search (heuristics like 2-Opt and 3-Opt). This hybrid approach is implemented using a sophisticated **Two-Level Island Model** for managing population diversity and preventing premature convergence.

The application is built in Java and uses the Spring Framework to manage data and provide real-time visualization.

## 🚀 Python + CUDA port (work in progress)

A ground-up rewrite of this solver in **Python + CUDA** (Numba) is under way in
[`pytsp/`](pytsp/) (branch `python-cuda-port`). It keeps the memetic algorithm but pushes the
**entire** evolutionary loop — including inter-island migration — onto the GPU using
**cooperative groups** (`grid.sync()`): the whole run executes inside a single *persistent
cooperative kernel*, and the CPU only launches it and reads back the best tour. There are
**no per-epoch GPU↔CPU round-trips** (the Java/Aparapi version returns to the CPU every epoch
for migration, tabu-list construction and integrity repair).

### What changed vs the Java / Aparapi version

| aspect | Java + Aparapi (original) | Python + CUDA (this port) |
|:--|:--|:--|
| **Per-epoch sync** | kernel is relaunched every epoch; the CPU does colony merge, Tabu-BST construction, integrity repair and selection between launches → a **GPU↔CPU round-trip every epoch** | one **persistent cooperative kernel**; a `grid.sync()` barrier per epoch; the CPU only launches once and reads back the best tour — **no round-trips** |
| **Local search** | random-sampling 2-opt / 3-opt / relocation (best of *K* random index triples) | **KNN neighbor-list** 2-opt / Or-opt / 3-opt with a position array and distance pruning — `O(n·K)` per sweep to a true local optimum |
| **Population** | 512 threads × 4 = **2 048** individuals (few GPU blocks) | 4 096–8 192 islands × 4 = **16k–32k** individuals (fills the GPU) |
| **Migration / merge** | on the CPU (gather, sort, power-law resample, merge colonies at time cutoffs) | **on the GPU**, race-free (migrant buffer + two-phase `grid.sync`), with global-merge windows |
| **Selection** | CPU sorts each colony, power-law fitness sampling | on-GPU **elitist steady-state** (a child displaces the weakest island member) |
| **Tabu** | CPU balanced BST of recurring tour lengths, ×1.004 penalty | **GPU age-penalty** (a leader that stays too long gets +0.4 % effective length) + a **tabu-immune best-ever tracker** so the true optimum is never lost |
| **Integrity** | operators can corrupt tours → GPU check + **CPU repair** every epoch | **permutation-preserving** operators — no repair step (validated on every run) |
| **Init / distances** | partial greedy | nearest-neighbor greedy; TSPLIB **EUC_2D / GEO / ATT** (so reported optima match) |

### Why the results improved so much

Three changes compound:

1. **No CPU round-trips.** The whole evolution — including migration, merge and tabu — stays
   on the GPU behind `grid.sync()`, so the GPU never idles waiting for the CPU between epochs.
   In the Java version every epoch returned to the CPU for migration, Tabu-BST and integrity
   repair; that serialization is gone.
2. **Neighbor-list local search.** Systematically probing each city's nearest neighbours
   (with distance pruning) reaches far stronger local optima than random index sampling —
   cheaply — and, unlike the random `O(n²)`-style sampling, it **scales to n > 3000**.
3. **Many more islands.** Extra islands fill otherwise-idle SMs almost for free at small *n*
   (measured **200 W → ~290 W, 100 % utilisation** on an RTX 3090), buying more search per
   wall-second.

Plus smaller refinements: parents are never re-optimised (they are already local-optimal —
only new children and perturbed tours get local search), a **double-bridge (4-opt)**
perturbation gives each island an ILS-style kick, and correct-by-construction operators
remove the integrity-repair round-trip entirely.

Net effect on the tested instances: **exact optima on small/mid problems, and several-fold
lower gaps in ~40 % of the original wall-clock time on large ones.**

**Early benchmark (RTX 3090)** — same TSPLIB optima, port vs original:

| instance | optimal | Java/Aparapi | **Python + CUDA** |
|:--|:--|:--|:--|
| berlin52, kroA100 | — | optimal | **optimal (0.000%)** |
| gr431 (GEO) | 171 414 | 0.64% / 212 s | **0.000% / 85 s** |
| pr1002 (EUC) | 259 045 | 1.19% / 817 s | **0.191% / 328 s** |

On the tested instances the port already **beats the original in both solution quality and
wall-clock time** — exact optimum on gr431 in ~40% of the time, and ~6× lower gap on pr1002,
also in ~40% of the time.

**A/B validation (controlled, same budget, RTX 3090).** The two optional diversity
mechanisms were each checked with an A/B run; both give a small, consistent gain on large
instances and are off by default (switchable):

| mechanism | instance | off | on |
|:--|:--|:--|:--|
| **colony merge** — global mixing at 4 budget checkpoints (0.25/0.5/0.75/0.9) | pr1002 (n=1002) | 0.191 % | **0.173 %** |
| **tabu** — a leader that stays too long in front gets a +0.4 % effective-length penalty in selection (its true best is kept in a tabu-immune tracker) | pcb3038 (n=3038, 300 epochs) | 1.458 % | **1.319 %** |

At the same budget pcb3038 already reaches **1.32 %** vs the original's 2.5–3.1 %. Scaling the
full `n > 3000` set is in progress.

> The Python + CUDA port was developed together with **Claude (Fable 5)** running in **Claude Code**.

---

# 🧬 The original Java + Aparapi solver

*Everything below documents the original Java + Aparapi implementation that the CUDA port above is based on.*

## Core Features

* **Hybrid Memetic Algorithm**: Fuses Genetic Algorithm operators (crossover, mutation) with powerful local search heuristics (2-Opt, 3-Opt, Segment Relocation) for rapid optimization.
* **Massive GPGPU Acceleration**: The entire evolutionary loop—including selection, crossover, and all memetic operators—runs in parallel on thousands of GPU threads via a custom Aparapi kernel (`TspGAKernel`).
* **Two-Level Island/Colony Model**: A sophisticated population structure that combines thousands of fast-evolving GPU "islands" with dozens of strategic CPU-managed "colonies" to ensure a robust global search.
* **Tabu Search Integration**: Employs a high-speed, BST-based Tabu list on the GPU to penalize recently visited solutions, helping the algorithm escape local optima.
* **Dynamic Problem Solving**: Automatically reads and solves `.tsp` files from a directory, logging all results.
* **Real-Time Visualization**: A built-in web server at `http://127.0.0.1:8080` visualizes the optimization process and the best path found in real-time.

## Algorithm Architecture: A Two-Level Hybrid Model

The solver's architecture is its most innovative feature. It divides the optimization problem into two distinct levels: a strategic CPU "Orchestrator" and a tactical GPU "Worker."

### Level 1 (CPU): The "Colony" Orchestrator
The main Java (Spring) application acts as the high-level strategist. Its responsibilities are:

* **Colony Management**: Divides the total population into a small number (`colonyMultiplier`) of large "Colonies."
* **Migration (Merging)**: Manages gene flow between these colonies. Most of the time, colonies evolve separately. Periodically (`shouldMergeColonies`), the CPU merges them into one "super-colony" to mix the best genetic material before splitting them apart again.
* **Tabu List Generation**: Analyzes results from all colonies and constructs a balanced Binary Search Tree (`bstTable`) of "tabu" (forbidden) solutions.
* **I/O and Control**: Handles file loading, result saving, and the visualization web server.
* **Adaptive Strategy**: Detects stagnation (`totalUnique < gpuThreads / 4`) and triggers a "Phase 2," dynamically changing the algorithm's parameters (e.g., enabling crossover, increasing population) to intensify the search.

### Level 2 (GPU): The "Island" Kernel
The custom `TspGAKernel` performs the computationally massive "tactical" work.

* **Island-per-Thread**: Each of the thousands of GPU threads (`gid`) acts as an independent, isolated "island."
* **Memetic Evolution**: Each "island" takes a tiny population (`pathsPerThread`) and runs a full, high-speed memetic algorithm on it in isolation for `epochsInGPU` generations.
* **Self-Contained Engine**: The kernel is entirely self-sufficient, containing its own GPU-safe random number generator (PRNG) and all memetic operators.
* **Local Optimization**: This is where the memetic "learning" happens. Each island intensely optimizes its local solutions using 2-Opt, 3-Opt, and other heuristics before reporting its best result back to the CPU orchestrator.

This two-level model allows the algorithm to simultaneously explore thousands of different solution paths in parallel (on GPU islands) while maintaining high-level strategic diversity (via CPU colonies).

## The Memetic Engine: Operators

The algorithm's power comes from its hybrid set of operators, which are all implemented to run directly on the GPU.

### Genetic Operators (Global Exploration)

* **Order Crossover (OX) (`crossOX`)**: The primary genetic operator. It creates new child routes by combining segments from two parent routes while preserving the relative order of cities, which is essential for TSP.

### Local Search Heuristics (Local Exploitation)

This is the "memetic" part. After a genetic operation, individuals are *immediately* improved using these powerful heuristics:

* **2-Opt (`mutTwoOpt`)**: The classic TSP heuristic. It finds two crossing edges in a route and reverses the segment between them to "uncross" them and shorten the path.
* **3-Opt (`mutThreeOpt`)**: A more powerful (and complex) version of 2-Opt. It removes three edges and tests all possible non-crossing reconnections to find the best improvement.
* **Relocation Operators**: A family of "move" operators that shift cities to new positions:
    * `mutSegmentRelocation`: Moves an entire segment (sub-path) to a new location.
    * `mutSingle/Two/ThreeVerticesRelocation`: Relocates 1, 2, or 3 adjacent cities to a new part of the route.
* **Swap Mutation (`mutVertexSwap`)**: A simple mutation that swaps the positions of two random cities.

## Performance & Robustness Features

* **GPU-Optimized Tabu Search**: The Tabu list is not a simple array. It's a balanced Binary Search Tree (`createBst`) passed to the GPU, allowing thousands of threads to query it (`searchInBst`) in parallel with high efficiency.
* **Custom GPU PRNG**: A custom, thread-safe pseudo-random number generator (based on XORShift) is implemented in the kernel (`random01()`). This is critical for high-performance GPGPU as it avoids the massive bottleneck of using `Math.random()`.
* **Integrity & Validation**: The kernel has a built-in `checkIntegrity` method to validate that routes are not corrupted during crossover/mutation. The CPU orchestrator (`checkAndRepairIntegrity`) then repairs any invalid routes reported by the GPU, ensuring 100% solution validity.

## Configuration

The project uses an `application.properties` configuration file. Below are key configuration options:

```properties
tsp.filename=full.tsp         # The problem file name
tsp.gpuThreads=512            # The number of GPU threads to be used
tsp.colonyMultiplier=4        # The number of colonies to divide the population into
tsp.divideGreedy=10           # The divisor for the greedy algorithm to initialize the population
tsp.scaleTime=0.01            # Time scaling factor for computation duration
tsp.mergeColonyByTime=true    # Enable merging of colonies based on elapsed time
tsp.cutoffsByTime=0.4,0.65,0.82,0.95  # Time points (as a fraction of total time) when colonies will merge
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/SebastianGruza/TspSolver.git
   ```

2. **Install required dependencies**:

   Ensure you have Java and Maven installed. Install any additional dependencies as specified in the `pom.xml` file.

3. **Build the project**:

   ```bash
   mvn clean install
   ```

4. **Configure the application**:

   Edit the `application.properties` file to set your desired configuration options.

5. **Run the application**:

   ```bash
   java -jar target/tsp-solver-1.0.jar
   ```

## Usage

1. **Prepare the TSP problem file**:

   - Place your `.tsp` problem file in the project folder.
   - Ensure the filename matches the `tsp.filename` property in `application.properties`.

2. **Configure parameters**:

   - Adjust settings in `application.properties` to suit your needs (e.g., GPU threads, colony multiplier).

3. **Start the TSP solving process**:

   - Run the application using the command provided in the installation section.

4. **Visualize the solution**:

   - Open your web browser and navigate to [http://127.0.0.1:8080](http://127.0.0.1:8080) to see the current best-known solution and real-time computation progress.

## Results and Performance

- **Scalability**: The application is designed to handle large TSP instances efficiently by leveraging GPU acceleration and advanced GA techniques.
- **Quality of Solutions**: Through the use of multiple mutation operators and adaptive strategies, the algorithm consistently finds high-quality solutions.
- **Performance Metrics**: Detailed logs and results are saved to `results.txt`, allowing you to analyze the algorithm's performance over time.

### Some results (TSP_LIB)

|problem|received|optimal|vertices|how_much_worse_than_optimum|time_in_seconds|
|:----|:----|:----|:----|:----|:----|
|burma14|3323|3323|14|0,0000%|5|
|burma14|3323|3323|14|0,0000%|5|
|ulysses16|6859|6859|16|0,0000%|5|
|ulysses22|7013|7013|22|0,0000%|5|
|att48|10628|10628|48|0,0000%|8|
|att48|10628|10628|48|0,0000%|8|
|eil51|426|426|51|0,0000%|9|
|eil51|426|426|51|0,0000%|9|
|berlin52|7542|7542|52|0,0000%|9|
|berlin52|7542|7542|52|0,0000%|9|
|st70|675|675|70|0,0000%|13|
|st70|675|675|70|0,0000%|13|
|eil76|538|538|76|0,0000%|14|
|pr76|108159|108159|76|0,0000%|14|
|eil76|538|538|76|0,0000%|14|
|pr76|108159|108159|76|0,0000%|14|
|gr96|55209|55209|96|0,0000%|19|
|gr96|55291|55209|96|0,1485%|19|
|rat99|1211|1211|99|0,0000%|20|
|rat99|1211|1211|99|0,0000%|20|
|kroA100|21282|21282|100|0,0000%|21|
|kroB100|22141|22141|100|0,0000%|21|
|kroC100|20749|20749|100|0,0000%|21|
|kroD100|21294|21294|100|0,0000%|21|
|kroE100|22106|22068|100|0,1722%|21|
|rd100|7910|7910|100|0,0000%|21|
|kroA100|21282|21282|100|0,0000%|21|
|kroB100|22141|22141|100|0,0000%|21|
|kroC100|20749|20749|100|0,0000%|21|
|kroD100|21309|21294|100|0,0704%|21|
|kroE100|22068|22068|100|0,0000%|21|
|rd100|7910|7910|100|0,0000%|21|
|eil101|629|629|101|0,0000%|21|
|eil101|629|629|101|0,0000%|21|
|lin105|14379|14379|105|0,0000%|22|
|lin105|14379|14379|105|0,0000%|22|
|pr107|44566|44303|107|0,5936%|23|
|pr107|44303|44303|107|0,0000%|23|
|pr124|59030|59030|124|0,0000%|28|
|pr124|59030|59030|124|0,0000%|28|
|bier127|118282|118282|127|0,0000%|29|
|bier127|118282|118282|127|0,0000%|29|
|ch130|6110|6110|130|0,0000%|30|
|ch130|6110|6110|130|0,0000%|30|
|pr136|96785|96772|136|0,0134%|33|
|pr136|96957|96772|136|0,1912%|33|
|gr137|69853|69853|137|0,0000%|33|
|gr137|69853|69853|137|0,0000%|33|
|pr144|58590|58537|144|0,0905%|36|
|pr144|58537|58537|144|0,0000%|36|
|ch150|6528|6528|150|0,0000%|38|
|kroA150|26550|26524|150|0,0980%|38|
|kroB150|26132|26130|150|0,0077%|38|
|ch150|6549|6528|150|0,3217%|38|
|kroA150|26525|26524|150|0,0038%|38|
|kroB150|26130|26130|150|0,0000%|38|
|pr152|74249|73682|152|0,7695%|39|
|pr152|73818|73682|152|0,1846%|39|
|u159|42080|42080|159|0,0000%|42|
|u159|42080|42080|159|0,0000%|42|
|rat195|2328|2323|195|0,2152%|58|
|rat195|2336|2323|195|0,5596%|58|
|d198|15780|15780|198|0,0000%|60|
|d198|15784|15780|198|0,0253%|60|
|kroA200|29368|29368|200|0,0000%|61|
|kroB200|29479|29437|200|0,1427%|61|
|kroA200|29451|29368|200|0,2826%|61|
|kroB200|29489|29437|200|0,1766%|61|
|gr202|40457|40160|202|0,7395%|62|
|gr202|40617|40160|202|1,1379%|62|
|ts225|126643|126643|225|0,0000%|73|
|tsp225|3921|3916|225|0,1277%|73|
|ts225|126643|126643|225|0,0000%|73|
|tsp225|3916|3916|225|0,0000%|73|
|pr226|80729|80369|226|0,4479%|74|
|pr226|80745|80369|226|0,4678%|74|
|gr229|135073|134602|229|0,3499%|76|
|gr229|135073|134602|229|0,3499%|76|
|gil262|2394|2378|262|0,6728%|94|
|gil262|2391|2378|262|0,5467%|94|
|pr264|49135|49135|264|0,0000%|95|
|pr264|49135|49135|264|0,0000%|95|
|a280|2579|2579|280|0,0000%|105|
|a280|2579|2579|280|0,0000%|105|
|pr299|48223|48191|299|0,0664%|117|
|pr299|48191|48191|299|0,0000%|117|
|lin318|42343|42029|318|0,7471%|129|
|linhp318|42149|41345|318|1,9446%|129|
|lin318|42265|42029|318|0,5615%|129|
|linhp318|42203|41345|318|2,0752%|129|
|rd400|15402|15281|400|0,7918%|188|
|rd400|15488|15281|400|1,3546%|188|
|fl417|11861|11861|417|0,0000%|201|
|fl417|11861|11861|417|0,0000%|201|
|gr431|172510|171414|431|0,6394%|212|
|gr431|173475|171414|431|1,2024%|212|
|pr439|108511|107217|439|1,2069%|219|
|pr439|107328|107217|439|0,1035%|219|
|pcb442|50938|50778|442|0,3151%|221|
|pcb442|50970|50778|442|0,3781%|221|
|d493|35293|35002|493|0,8314%|264|
|d493|35257|35002|493|0,7285%|264|
|att532|27911|27686|532|0,8127%|298|
|att532|28047|27686|532|1,3039%|298|
|ali535|203586|202339|535|0,6163%|301|
|ali535|203160|202339|535|0,4058%|301|
|u574|37410|36905|574|1,3684%|337|
|rat575|6843|6773|575|1,0335%|338|
|rat575|6863|6773|575|1,3288%|338|
|p654|34649|34643|654|0,0173%|416|
|p654|34643|34643|654|0,0000%|416|
|d657|49708|48912|657|1,6274%|419|
|d657|49543|48912|657|1,2901%|419|
|gr666|298335|294358|666|1,3511%|428|
|gr666|298552|294358|666|1,4248%|428|
|u724|42379|41910|724|1,1191%|489|
|rat783|8917|8806|783|1,2605%|554|
|rat783|8900|8806|783|1,0675%|554|
|pr1002|262117|259045|1002|1,1859%|817|
|pr1002|264951|259045|1002|2,2799%|817|
|u1060|227440|224094|1060|1,4931%|892|
|u1060|227431|224094|1060|1,4891%|892|
|vm1084|242112|239297|1084|1,1764%|924|
|pcb1173|58431|56892|1173|2,7051%|1046|
|pcb1173|58057|56892|1173|2,0477%|1046|
|d1291|51406|50801|1291|1,1909%|1214|
|d1291|51008|50801|1291|0,4075%|1214|
|rl1304|257212|252948|1304|1,6857%|1233|
|rl1304|254771|252948|1304|0,7207%|1233|
|rl1323|275161|270199|1323|1,8364%|1261|
|rl1323|276402|270199|1323|2,2957%|1261|
|nrw1379|57831|56638|1379|2,1064%|1344|
|nrw1379|57617|56638|1379|1,7285%|1344|
|fl1400|20317|20127|1400|0,9440%|1376|
|fl1400|20180|20127|1400|0,2633%|1376|
|u1432|155615|152970|1432|1,7291%|1425|
|u1432|156243|152970|1432|2,1396%|1425|
|fl1577|22278|22249|1577|0,1303%|1654|
|fl1577|22291|22249|1577|0,1888%|1654|
|d1655|62522|62128|1655|0,6342%|1781|
|d1655|62761|62128|1655|1,0189%|1781|
|vm1748|340416|336556|1748|1,1469%|1937|
|u1817|58153|57201|1817|1,6643%|2055|
|rl1889|324426|316536|1889|2,4926%|2181|
|rl1889|322750|316536|1889|1,9631%|2181|
|d2103|81041|80450|2103|0,7346%|2568|
|d2103|80982|80450|2103|0,6613%|2568|
|u2152|65544|64253|2152|2,0092%|2660|
|u2319|236815|234256|2319|1,0924%|2979|
|pr2392|385518|378032|2392|1,9803%|3122|
|pr2392|385489|378032|2392|1,9726%|3122|
|pcb3038|141952|137694|3038|3,0924%|4473|
|pcb3038|141131|137694|3038|2,4961%|4473|
|fl3795|29223|28772|3795|1,5675%|6233|
|fl3795|29055|28772|3795|0,9836%|6233|
|fnl4461|187755|182566|4461|2,8423%|7917|
|fnl4461|187081|182566|4461|2,4731%|7917|
|rl5915|578560|565530|5915|2,3040%|11978|
|rl5915|578771|565530|5915|2,3413%|11978|
|rl5934|570834|556045|5934|2,6597%|12034|
|rl5934|568102|556045|5934|2,1683%|12034|
|gr9882|313656|300899|9882|4,2396%|25172|
|gr9882|311626|300899|9882|3,5650%|25172|


### Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow the guidelines outlined in `CONTRIBUTING.md`.

### License

This project is licensed under the [MIT License](LICENSE).
