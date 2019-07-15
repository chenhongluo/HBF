--------------------------------------------
REQUIREMENTS
--------------------------------------------

CMake version >= 30
CUDA version >= 7
NVIDIA Kepler or Maxwell GPU with compute capability >= 35
[ Optional ] Boost Library to test Boost implementations of Dijkstra and Bellman-Ford Algorithms. 

--------------------------------------------
COMPILE
--------------------------------------------

$ cd build
$ cmake -DARCH=<your_compute_cabability> -DSM=<number_of_streaming_multiprocessor> ..
$ make -j

example:
$ cmake -DARCH=35 -DSM=12 ..

--------------------------------------------
USAGE
--------------------------------------------

$ ./HBF <graph_path>

config.cuh	-> advanced HBF configuration

--------------------------------------------
SUPPORTED INPUT FORMAT
--------------------------------------------

SNAP, METIS, GTGRAPH, MATRIX MARKET (mtx), DIMACS9TH, DIMACS10TH

see GraphFormat.txt for more information


rm -r * 

cmake -DARCH=35 -DSM=68 .. 

make -j

./HBF /home/chl/data/flickr.mtx >>log.txt

