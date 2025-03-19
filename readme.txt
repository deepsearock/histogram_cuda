Lab2: histogram

Zhaorui Wang
cwid: 20007447

The lab and spreadsheet with some data is in here aswell as groovy extension program with some pseudo code for questions in assignment

In order to build you must run

mkdir build && cd build
cmake ..
cmake --build .

This will yield 4 programs
./comparison
./graph
./histogram

./comparison utilization:
Usage: ./comparison -i <BinNum> <VecDim> <GridSize> this runs the main program which compares tiled vs optimized vs naive kernels
  <BinNum>: is the number of bins (any integer that can be written as 2^k where k can be any integer from 2 to 8).
  <VecDim>: is the dimension of the input vector
  <GridSize>: gridsize which is used for tiled and optimized kernel should input some value above 256, I use 256,512,1024

./histogram -i <BinNum> <VecDim> <GridSize> this runs just the optimized kernel.
  <BinNum>: is the number of bins (any integer that can be written as 2^k where k can be any integer from 2 to 8).
  <VecDim>: is the dimension of the input vector
  <GridSize>: gridsize which is used for tiled and optimized kernel should input some value above 256, I use 256,512,1024

./graph -i <VecDim> <GridSize> this program runs through all bins 4,8,16,32,64,128,256 and returns the values in a csv file.
  <VecDim>: Number of columns in matrix A (and number of rows in matrix B)
  <GridSize>: Number of columns in matrix B and matrix C
