# EDA Automation – Partitioning Stage

This repository implements two heuristic algorithms for hypergraph partitioning: **Kernighan–Lin (KL)** and **Fiduccia–Mattheyses (FM)**. Both algorithms are provided in serial, OpenMP-parallelized, and CUDA-accelerated versions.

---

## Project Overview

These algorithms aim to minimize **cut size** in hypergraph partitioning problems, a core task in VLSI design automation. The implementations are optimized for different hardware platforms to study performance and scaling behavior:

- **KL (Serial, OpenMP, CUDA)**
- **FM (Serial, OpenMP, CUDA)**

---

## How to Compile

Ensure you have:
- `g++` with OpenMP support
- NVIDIA CUDA toolkit (for GPU versions)
- `make`

Then, compile all targets with:

```bash
make
```
To run the compiled codes with input files from the circuits folder
```bash
./fm_cuda circuits/ibm01.hgr
./kl_cuda circuits/ibm01.hgr
./kl circuits/ibm01.hgr
./fm circuits/ibm01.hgr
./klomp circuits/ibm01.hgr
./fmomp circuits/ibm01.hgr
```

The FM also has additional flags, if not set they take the default values for number of passes, threads to use and max diff. Providing an example with these flags below
The example below does a max of 5 passes with 2 nodes max difference between both partitions and openmp threads of 4.
```bash
./fmomp 5 2 -t 4 circuits/ibm01.hgr
```
