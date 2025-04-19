#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <unordered_map>

using namespace std;

// Simple CUDA error‐checker
inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA error at " << msg << ": " 
             << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

// -----------------------------------------------------------------------------
// Device kernels
// -----------------------------------------------------------------------------
__global__
void ComputeGains(int N,
                  const int* rowPtr,
                  const int* colIdx,
                  const float* weights,
                  const int* partition,
                  const unsigned char* locked,
                  float* nodeGains)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (locked[i]) {
        nodeGains[i] = INFINITY;  // never pick again
        return;
    }

    int side = partition[i];
    float inW = 0.0f, extW = 0.0f;
    for (int e = rowPtr[i]; e < rowPtr[i+1]; ++e) {
        int j = colIdx[e];
        if (partition[j] == side) inW  += weights[e];
        else                       extW += weights[e];
    }
    nodeGains[i] = extW - inW;
}

__global__
void ComputeCutContrib(int N,
                       const int* rowPtr,
                       const int* colIdx,
                       const float* weights,
                       const int* partition,
                       float* cutContrib)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum = 0.0f;
    for (int e = rowPtr[i]; e < rowPtr[i+1]; ++e) {
        if (partition[i] != partition[colIdx[e]])
            sum += weights[e];
    }
    cutContrib[i] = sum;
}

__global__
void ApplySwap(int n1,
               int n2,
               int* partition,
               unsigned char* locked)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int tmp = partition[n1];
        partition[n1] = partition[n2];
        partition[n2] = tmp;
        locked[n1] = locked[n2] = 1;
    }
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
string getBaseName(const string& path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == string::npos)
         ? path
         : path.substr(pos + 1);
}

void createDir(const string& dir) {
    struct stat info;
    if (stat(dir.c_str(), &info) != 0) {
        mkdir(dir.c_str(), 0755);
    }
}

// Build a full, symmetric CSR from upper‐triangle adjacency
void buildCSR(const vector<unordered_map<int,float>>& adj,
              int N,
              vector<int>& rowPtr,
              vector<int>& colIdx,
              vector<float>& weights)
{
    vector<vector<pair<int,float>>> full(N);
    for (int i = 0; i < N; ++i) {
        for (auto& kv : adj[i]) {
            int j = kv.first; float w = kv.second;
            full[i].emplace_back(j, w);
            full[j].emplace_back(i, w);
        }
    }
    rowPtr.resize(N+1);
    rowPtr[0] = 0;
    for (int i = 0; i < N; ++i)
        rowPtr[i+1] = rowPtr[i] + int(full[i].size());
    int nnz = rowPtr[N];
    colIdx .resize(nnz);
    weights.resize(nnz);
    for (int i = 0; i < N; ++i) {
        int idx = rowPtr[i];
        for (auto& p : full[i]) {
            colIdx[idx]   = p.first;
            weights[idx++] = p.second;
        }
    }
}

float getEdgeWeight(const vector<unordered_map<int,float>>& adj,
                    int u, int v)
{
    if (u > v) swap(u,v);
    auto it = adj[u].find(v);
    return (it != adj[u].end()) ? it->second : 0.0f;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_file.hgr>\n";
        return 1;
    }
    string infile = argv[1];
    string base   = getBaseName(infile);
    createDir("results");

    ofstream fout("results/" + base + "_KL_CutSize_output.txt");
    if (!fout.is_open()) {
        cerr << "Error creating results file\n";
        return 1;
    }

    // --- 1) Read .hgr into an upper‐triangle adjacencyList ---
    ifstream fin(infile);
    if (!fin.is_open()) {
        cerr << "Error opening " << infile << "\n";
        return 1;
    }
    long netsNum, nodesNum;
    {
        string header;
        getline(fin, header);
        stringstream ss(header);
        ss >> netsNum >> nodesNum;
    }
    int N = int(nodesNum);
    vector<unordered_map<int,float>> adj(N);
    vector<int>    temp;
    for (int i = 0; i < netsNum; ++i) {
        string line; getline(fin, line);
        stringstream ss(line);
        temp.clear();
        int x;
        while (ss >> x) temp.push_back(x-1);
        if (temp.size() < 2) continue;
        float w = 1.0f / float(temp.size()-1);
        for (size_t a = 0; a < temp.size(); ++a)
            for (size_t b = a+1; b < temp.size(); ++b) {
                int u = temp[a], v = temp[b];
                if (u>v) swap(u,v);
                adj[u][v] += w;
            }
    }
    fin.close();

    // --- 2) Build symmetric CSR for GPU ---
    vector<int>   h_rowPtr, h_colIdx;
    vector<float> h_weights;
    buildCSR(adj, N, h_rowPtr, h_colIdx, h_weights);

    // --- 3) Initial, balanced random partition on CPU ---
    vector<int>           h_part(N);
    vector<unsigned char> h_locked(N, 0);
    {
        vector<int> perm(N);
        iota(perm.begin(), perm.end(), 0);
        mt19937 rng(random_device{}());
        shuffle(perm.begin(), perm.end(), rng);
        int half = N/2;
        for (int i = 0; i < N; ++i)
            h_part[perm[i]] = (i < half ? 0 : 1);
    }

    // --- 4) Allocate + copy to GPU ---
    int *d_rowPtr, *d_colIdx, *d_part;
    float *d_weights, *d_gains, *d_cutC;
    unsigned char *d_locked;
    checkCuda(cudaMalloc(&d_rowPtr, (N+1)*sizeof(int)),          "malloc rowPtr");
    checkCuda(cudaMalloc(&d_colIdx, h_colIdx.size()*sizeof(int)), "malloc colIdx");
    checkCuda(cudaMalloc(&d_weights, h_weights.size()*sizeof(float)), "malloc weights");
    checkCuda(cudaMalloc(&d_part,   N*sizeof(int)),               "malloc part");
    checkCuda(cudaMalloc(&d_gains,  N*sizeof(float)),             "malloc gains");
    checkCuda(cudaMalloc(&d_cutC,   N*sizeof(float)),             "malloc cutC");
    checkCuda(cudaMalloc(&d_locked, N*sizeof(unsigned char)),     "malloc locked");

    checkCuda(cudaMemcpy(d_rowPtr, h_rowPtr.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice), "copy rowPtr");
    checkCuda(cudaMemcpy(d_colIdx, h_colIdx.data(), h_colIdx.size()*sizeof(int), cudaMemcpyHostToDevice), "copy colIdx");
    checkCuda(cudaMemcpy(d_weights,h_weights.data(), h_weights.size()*sizeof(float), cudaMemcpyHostToDevice),   "copy weights");
    checkCuda(cudaMemcpy(d_part,   h_part.data(),   N*sizeof(int), cudaMemcpyHostToDevice), "copy part");
    checkCuda(cudaMemcpy(d_locked, h_locked.data(), N*sizeof(unsigned char), cudaMemcpyHostToDevice), "copy locked");

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    // start GPU KL timer
    auto start = chrono::steady_clock::now();

    // --- 5) Compute initial cut size on GPU ---
    ComputeCutContrib<<<blocks,threads>>>(N, d_rowPtr, d_colIdx, d_weights, d_part, d_cutC);
    checkCuda(cudaDeviceSynchronize(), "init cut sync");
    thrust::device_ptr<float> dc_ptr = thrust::device_pointer_cast(d_cutC);
    float total = thrust::reduce(dc_ptr, dc_ptr + N, 0.0f);
    float cutSize = total * 0.5f;
    float initial = cutSize;
    fout << "0\t" << cutSize << "\t0\n";

    // --- 6) KL iterations ---
    int iteration = 0;
    int negativeStreak = 0;
    int limit = int(log2f(N)) + 5;

    while (true) {
        // 6a) Compute gains
        ComputeGains<<<blocks,threads>>>(N, d_rowPtr, d_colIdx, d_weights, d_part, d_locked, d_gains);
        checkCuda(cudaDeviceSynchronize(), "gains sync");

        // 6b) Copy gains back
        vector<float> h_gains(N);
        checkCuda(cudaMemcpy(h_gains.data(), d_gains, N*sizeof(float), cudaMemcpyDeviceToHost), "copy gains");

       
        // 1) Find the highest‐gain candidate in **both** parts
        float best0 = -INFINITY, best1 = -INFINITY;
        int n1 = -1, n2 = -1;
        for (int i = 0; i < N; ++i) {
        if (!h_locked[i]) {
            if (h_part[i] == 0 && h_gains[i] > best0) {
            best0 = h_gains[i];
            n1    = i;
            }
            else if (h_part[i] == 1 && h_gains[i] > best1) {
            best1 = h_gains[i];
            n2    = i;
            }
        }
        }
        if (n1 < 0 || n2 < 0) break;

        // 2) True KL gain formula
        float w12  = getEdgeWeight(adj, n1, n2);
        float gain = best0 + best1 - 2.0f * w12;
        if (gain <= 0 && ++negativeStreak > limit) break;
        if (gain > 0) negativeStreak = 0;

        // 1) Update your host‐side arrays so you can continue scanning h_part/h_locked:
        h_part  [n1]   = 1;
        h_part  [n2]   = 0;
        h_locked[n1]   = 1;
        h_locked[n2]   = 1;

        // 2) Let the GPU do the in‐place swap and lock
        ApplySwap<<<1,1>>>(n1, n2, d_part, d_locked);
        checkCuda(cudaDeviceSynchronize(), "swap sync");

        // 3) Recompute cut on GPU
        ComputeCutContrib<<<blocks,threads>>>(N,
            d_rowPtr, d_colIdx, d_weights,
            d_part,
            d_cutC);
        checkCuda(cudaDeviceSynchronize(), "cut sync");

        // 4) Sum and halve
        float total = thrust::reduce(dc_ptr, dc_ptr + N, 0.0f);
        cutSize = total * 0.5f;

        // 5) Record iteration
        iteration++;
        fout << iteration << "\t" << cutSize << "\t" << gain << "\n";
        cout     << "Iter " << iteration
                << "  cut=" << cutSize
                << "  gain=" << gain << "\n";
    }

    // stop timer and report execution time
    auto end = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Execution Time: " << elapsed << " ms" << endl;

    fout.close();
    cout << "Initial cut = " << initial
         << "\nFinal   cut = " << cutSize << "\n";

    // --- 7) Cleanup ---
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_weights);
    cudaFree(d_part);
    cudaFree(d_gains);
    cudaFree(d_cutC);
    cudaFree(d_locked);

    return 0;
}