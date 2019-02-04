The programs require Dynet (https://github.com/clab/dynet) installed, compiled for C++.
I have also been using Intel MKL, donwloadable freely, and built Dynet to use MKL. 
In my early testing CPU perf was better than GPU one, so did not used GPU builds of Dynet.
There will be 4 projects, each containing one .cc file and slstm.*.
The programs can be run on Windows, Linux, and Mac.
See inside *.cc files - there are more details. You need to setup some params.

I provide example scripts for Linux, and a VS 2015 solution for Windows.