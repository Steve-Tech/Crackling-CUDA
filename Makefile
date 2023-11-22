# the compiler:
CC = g++
NVCC = nvcc

# compiler flags:
CFLAGS = -O3 -std=c++11 -fopenmp -mpopcnt
NVCFLAGS = -O3 -std=c++11 -Xcompiler -fopenmp -Xcompiler -mpopcnt

# define any directories containing header files other than /usr/include
INCLUDES = -Isrc/ISSL/include

all : isslScoreOfftargets isslCreateIndex
cuda : isslScoreOfftargetsCUDA isslCreateIndex

isslScoreOfftargets : src/ISSL/isslScoreOfftargets.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -o bin/$@ $^

isslCreateIndex : src/ISSL/isslCreateIndex.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -o bin/$@ $^

isslScoreOfftargetsCUDA : src/ISSL/isslScoreOfftargets.cu
	$(NVCC) $(NVCFLAGS) $(INCLUDES) -o bin/$@ $^

# Not used yet
isslCreateIndexCUDA : src/ISSL/isslCreateIndex.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) -o bin/$@ $^

clean:
	$(RM) bin/isslScoreOfftargets bin/isslCreateIndex
