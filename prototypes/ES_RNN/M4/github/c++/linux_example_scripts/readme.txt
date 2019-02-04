build_mkl builds a specified program , linking it with MKL-compiled version of Dynet.
usage, e.g.:
./build_mkl ES_RNN
(no extension).
____You need to modify it, to point to your location of Dynet library.____
Also, remove -lodbc if you do not use it, and especially if you had not installed it :-)

run18 is a script that runs 9 pairs of workers, to be used with ES_RNN and ES_RNN_PI. 
So it assumes it runs on a nice 18-core machine :-), and in such case you BIG_LOOP constant in the .cc files should probably be = 1, no big need for more than 9 runs for assembling.
usage, e.g.:
./run18 ES_RNN


