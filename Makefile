# Makefile for FM and KL implementations (serial, OpenMP, and CUDA)

# Compilers
CXX = g++
NVCC = nvcc

# Flags
CXXFLAGS = -O3
OMPFLAGS = -fopenmp
NVFLAGS = -O3 -Xcompiler -fopenmp

# Targets
all: fm fmomp kl klomp kl_cuda fm_cuda

# FM Serial
fm: FM_omp.cpp
	$(CXX) $(CXXFLAGS) FM_omp.cpp -o fm

# FM OpenMP
fmomp: FM_omp.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) FM_omp.cpp -o fmomp

# KL Serial
kl: cKL.cpp
	$(CXX) $(CXXFLAGS) cKL.cpp -o kl

# KL OpenMP (same source as serial, but with OpenMP)
klomp: cKL.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) cKL.cpp -o klomp

# KL CUDA
kl_cuda: kl_cuda.cu
	$(NVCC) $(NVFLAGS) kl_cuda.cu -o kl_cuda

# FM CUDA
fm_cuda: FM_cuda.cu
	$(NVCC) $(NVFLAGS) FM_cuda.cu -o fm_cuda

# Clean
clean:
	rm -f fm fmomp kl klomp kl_cuda fm_cuda