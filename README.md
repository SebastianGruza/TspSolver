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
