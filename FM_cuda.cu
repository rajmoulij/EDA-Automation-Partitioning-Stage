// FM_cuda_v2_optimized_with_gain_gpu.cu

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
using namespace std;

// Global debug flag: Set to true to enable debug prints.
bool debug = false;

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) {                                    \
    cudaError_t err = call;                                   \
    if(err != cudaSuccess) {                                  \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                  << " " << cudaGetErrorString(err) << std::endl; \
        exit(err);                                            \
    }                                                         \
}
#endif

// ------------------------- Forward Declarations -------------------------
float gpuCalCutSizePersistent(const int* d_netNodes, const int* d_netOffsets,
                                const int* d_partitionAssignment, float* d_netCuts, int netsCount,
                                cudaStream_t stream, cudaEvent_t event);
void readAndPartition(const string &filename,
                      vector<vector<int>> &nets,
                      vector<int> &partition1,
                      vector<int> &partition2,
                      int &nodesCount);
void flattenNets(const vector<vector<int>>& nets, vector<int>& flatNetNodes, vector<int>& netOffsets);
void buildNodeIncidenceList(const vector<vector<int>>& nets, int nodesCount,
                            vector<int>& nodeNetList, vector<int>& nodeNetOffsets);
void printPartitionContent(const vector<int>& partitionAssignment);

// ------------------------- CUDA Kernel for Cut Size Calculation -------------------------
__global__
void computeCutSizeKernel(const int* d_netNodes, const int* d_netOffsets,
                          const int* d_partitionAssignment,
                          float* d_netCuts, int netsCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < netsCount) {
        int start = d_netOffsets[i];
        int end = d_netOffsets[i+1];
        int netSize = end - start;
        float localCost = 0.0f;
        if (netSize >= 2) {
            float weight = 1.0f / (netSize - 1);
            for (int j = start; j < end; j++) {
                for (int k = j + 1; k < end; k++) {
                    int node1 = d_netNodes[j];
                    int node2 = d_netNodes[k];
                    if (d_partitionAssignment[node1] != d_partitionAssignment[node2])
                        localCost += weight;
                }
            }
        }
        d_netCuts[i] = localCost;
    }
}

// Persistent GPU cut size function.
float gpuCalCutSizePersistent(const int* d_netNodes, const int* d_netOffsets,
                              const int* d_partitionAssignment, float* d_netCuts, int netsCount,
                              cudaStream_t stream, cudaEvent_t event) {
    int threadsPerBlock = 256;  // Adjust if needed.
    int blocks = (netsCount + threadsPerBlock - 1) / threadsPerBlock;
    computeCutSizeKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_netNodes, d_netOffsets,
                                                                 d_partitionAssignment,
                                                                 d_netCuts, netsCount);
    // Record and wait on the event.
    CUDA_CHECK(cudaEventRecord(event, stream));
    CUDA_CHECK(cudaEventSynchronize(event));
    
    vector<float> h_netCuts(netsCount);
    CUDA_CHECK(cudaMemcpyAsync(h_netCuts.data(), d_netCuts, netsCount * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    float totalCost = 0.0f;
    for (int i = 0; i < netsCount; i++) {
        totalCost += h_netCuts[i];
    }
    
    return totalCost;
}

// ------------------------- CUDA Kernel for Node Gain Calculation -------------------------
__global__
void computeNodeGainsKernel(const int* d_netNodes, const int* d_netOffsets,
                            const int* d_nodeNetList, const int* d_nodeNetOffsets,
                            const int* d_partitionAssignment,
                            float* d_nodeGains,
                            int numNodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < numNodes) {
        int myPart = d_partitionAssignment[node];
        float internal = 0.0f;
        float external = 0.0f;
        // Iterate over each net in which this node appears.
        for (int idx = d_nodeNetOffsets[node]; idx < d_nodeNetOffsets[node + 1]; idx++) {
            int netId = d_nodeNetList[idx];
            int start = d_netOffsets[netId];
            int end   = d_netOffsets[netId+1];
            // Iterate over all nodes in this net.
            for (int j = start; j < end; j++) {
                int other = d_netNodes[j];
                if (other == node)
                    continue;
                if (d_partitionAssignment[other] == myPart)
                    internal += 1.0f;
                else
                    external += 1.0f;
            }
        }
        d_nodeGains[node] = external - internal;
    }
}

// ------------------------- Host Data Structures -------------------------
// FM Node structure (host only)
struct FMNode {
    int id;         // Node identifier
    float gain;     // Gain value
    FMNode* next;   // Pointer to the next node in the doubly linked list
    FMNode* prev;   // Pointer to the previous node

    FMNode(int id, float gain) : id(id), gain(gain), next(nullptr), prev(nullptr) {}
};

// Doubly linked list for FM gains (host only)
class GainList {
private:
    FMNode* head;
    FMNode* tail;
public:
    GainList() : head(nullptr), tail(nullptr) {}
    ~GainList() { clear(); }
    
    void clear() {
        while (head) {
            FMNode* temp = head;
            head = head->next;
            delete temp;
        }
        tail = nullptr;
    }
    
    void insertNode(FMNode* newNode) {
        if (!head) {
            head = tail = newNode;
            return;
        }
        if (newNode->gain >= head->gain) {
            newNode->next = head;
            head->prev = newNode;
            head = newNode;
            return;
        }
        FMNode* current = head;
        while (current != nullptr && newNode->gain < current->gain) {
            current = current->next;
        }
        if (current == nullptr) {
            tail->next = newNode;
            newNode->prev = tail;
            tail = newNode;
        } else {
            newNode->prev = current->prev;
            newNode->next = current;
            if (current->prev)
                current->prev->next = newNode;
            current->prev = newNode;
        }
    }
    
    void printList() const {
        FMNode* current = head;
        while (current) {
            cout << "Node ID: " << current->id << ", Gain: " << current->gain << endl;
            current = current->next;
        }
    }
};

// ------------------------- Helper Functions on Host -------------------------
void printPartitionContent(const vector<int>& partitionAssignment) {
    if (!debug)
        return;
    int n = partitionAssignment.size();
    cout << "Current Partition Assignment:" << endl;
    cout << "Partition 0: ";
    for (int i = 0; i < n; i++) {
        if (partitionAssignment[i] == 0)
            cout << i << " ";
    }
    cout << "\nPartition 1: ";
    for (int i = 0; i < n; i++) {
        if (partitionAssignment[i] == 1)
            cout << i << " ";
    }
    cout << "\n--------------------------------\n";
}

void readAndPartition(const string &filename,
                      vector<vector<int>> &nets,
                      vector<int> &partition1,
                      vector<int> &partition2,
                      int &nodesCount) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "Error opening file: " << filename << endl;
        return;
    }
    int netsCount;
    string line;
    cout << "\n Starting Reading Input File \n";
    getline(infile, line);
    {
        stringstream ss(line);
        ss >> netsCount >> nodesCount;
    }
    nets.resize(netsCount);
    
    vector<string> netLines(netsCount);
    for (int i = 0; i < netsCount; i++) {
        getline(infile, netLines[i]);
    }
    infile.close();
    cout << "\n Completed Reading Input File \n";

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < netsCount; i++) {
        stringstream netStream(netLines[i]);
        int node;
        vector<int> netNodes;
        while (netStream >> node) {
            netNodes.push_back(node - 1);
        }
        nets[i] = move(netNodes);
    }
    
    vector<int> nodes(nodesCount);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nodesCount; i++) {
        nodes[i] = i;
    }
    
    random_device rd;
    mt19937 gen(rd());
    shuffle(nodes.begin(), nodes.end(), gen);
    
    int mid = nodesCount / 2;
    partition1.assign(nodes.begin(), nodes.begin() + mid);
    partition2.assign(nodes.begin() + mid, nodes.end());
    cout << "\n Completed Splitting parts \n";
}

// Flatten nets into a single array and compute offsets.
void flattenNets(const vector<vector<int>>& nets, vector<int>& flatNetNodes, vector<int>& netOffsets) {
    int netsCount = nets.size();
    cout << "\n Started Flattening Nets \n";
    // Phase 1: Compute each net's size in parallel.
    vector<int> netSizes(netsCount);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < netsCount; i++){
        netSizes[i] = nets[i].size();
    }
    
    // Phase 2: Compute prefix sum sequentially.
    netOffsets.resize(netsCount + 1);
    netOffsets[0] = 0;
    partial_sum(netSizes.begin(), netSizes.end(), netOffsets.begin() + 1);
    
    // Phase 3: Fill in flatNetNodes in parallel.
    flatNetNodes.resize(netOffsets[netsCount]);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < netsCount; i++) {
        int start = netOffsets[i];
        for (size_t j = 0; j < nets[i].size(); j++) {
            flatNetNodes[start + j] = nets[i][j];
        }
    }
    cout << "\n Completed Flattening Nets \n";
}

// Build node incidence list: for each node, record which nets it belongs to.
void buildNodeIncidenceList(const vector<vector<int>>& nets, int nodesCount,
                            vector<int>& nodeNetList, vector<int>& nodeNetOffsets) {
    vector<vector<int>> incidence(nodesCount);
    for (size_t netId = 0; netId < nets.size(); netId++) {
        for (int node : nets[netId]) {
            incidence[node].push_back(netId);
        }
    }
    // Compute offsets and flatten the incidence list.
    nodeNetOffsets.resize(nodesCount + 1);
    nodeNetOffsets[0] = 0;
    for (int i = 0; i < nodesCount; i++) {
        nodeNetOffsets[i+1] = nodeNetOffsets[i] + incidence[i].size();
    }
    nodeNetList.resize(nodeNetOffsets[nodesCount]);
    for (int i = 0; i < nodesCount; i++) {
        int offset = nodeNetOffsets[i];
        for (size_t j = 0; j < incidence[i].size(); j++) {
            nodeNetList[offset + j] = incidence[i][j];
        }
    }
}

// ------------------------- Modified FM Algorithm Pass Using GPU -------------------------

float FMAlgorithmPass(const vector<vector<int>>& nets, vector<int>& partitionAssignment,
    int nodesCount, int balanceThreshold,
    const int* d_netNodes, const int* d_netOffsets, int netsCountDevice,
    int* d_partitionAssignment, float* d_netCuts,
    // New parameters for node gain kernel:
    const int* d_nodeNetList, const int* d_nodeNetOffsets,
    float* d_nodeGains,
    cudaStream_t stream, cudaEvent_t event) {
// Create a vector to keep track of whether a node has been locked (moved)
vector<bool> locked(nodesCount, false);

// Copy the partitionAssignment array to the GPU.
CUDA_CHECK(cudaMemcpy(d_partitionAssignment, partitionAssignment.data(),
      nodesCount * sizeof(int), cudaMemcpyHostToDevice));

// Compute the initial cut size using GPU kernels.
float initialCutSize = gpuCalCutSizePersistent(d_netNodes, d_netOffsets, d_partitionAssignment, d_netCuts, 
                               netsCountDevice, stream, event);
float bestCutSize = initialCutSize;
vector<int> bestAssignment = partitionAssignment;

// Compute initial node gains on GPU.
int threadsPerBlock = 256;
int blocks = (nodesCount + threadsPerBlock - 1) / threadsPerBlock;
computeNodeGainsKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_netNodes, d_netOffsets,
                                               d_nodeNetList, d_nodeNetOffsets,
                                               d_partitionAssignment,
                                               d_nodeGains,
                                               nodesCount);
CUDA_CHECK(cudaStreamSynchronize(stream));

vector<float> nodeGains(nodesCount);
CUDA_CHECK(cudaMemcpy(nodeGains.data(), d_nodeGains, nodesCount * sizeof(float),
      cudaMemcpyDeviceToHost));

int moves = 0;
while (moves < nodesCount) {
// Perform reduction on the GPU using Thrust.
// Wrap the device pointer for the partition assignment.
thrust::device_ptr<int> devPartPtr(d_partitionAssignment);
int count0 = thrust::count(devPartPtr, devPartPtr + nodesCount, 0);
int count1 = nodesCount - count0; // Since the values are either 0 or 1.
int diff = count0 - count1;

float maxGain = -1e9;
int bestNode = -1;
// Loop over all nodes on the host to pick the best candidate,
// using the node gains computed on the GPU.
for (int i = 0; i < nodesCount; i++) {
if (!locked[i]) {
// Check the balance condition: if the difference exceeds the threshold,
// ensure that moving this node would adhere to it.
if (abs(diff) > balanceThreshold) {
if (diff > balanceThreshold && partitionAssignment[i] != 0)
    continue;
if (-diff > balanceThreshold && partitionAssignment[i] != 1)
    continue;
}
if (nodeGains[i] > maxGain) {
maxGain = nodeGains[i];
bestNode = i;
}
}
}
if (bestNode == -1)
break;

// Flip the partition assignment for the chosen node and mark it as locked.
partitionAssignment[bestNode] = 1 - partitionAssignment[bestNode];
locked[bestNode] = true;
moves++;

// Update the GPU copy of the partition assignment.
CUDA_CHECK(cudaMemcpy(d_partitionAssignment, partitionAssignment.data(),
          nodesCount * sizeof(int), cudaMemcpyHostToDevice));

float currentCutSize = gpuCalCutSizePersistent(d_netNodes, d_netOffsets, d_partitionAssignment, d_netCuts, 
                                   netsCountDevice, stream, event);
if (debug)
cout << "Move " << moves << " - Moved node " << bestNode 
<< ", Current cut size: " << currentCutSize << endl;
if (currentCutSize < bestCutSize) {
bestCutSize = currentCutSize;
bestAssignment = partitionAssignment;
if (debug)
cout << "New best cut size: " << bestCutSize << endl;
}

// Recompute node gains on GPU after the partition change.
computeNodeGainsKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_netNodes, d_netOffsets,
                                                   d_nodeNetList, d_nodeNetOffsets,
                                                   d_partitionAssignment,
                                                   d_nodeGains,
                                                   nodesCount);
CUDA_CHECK(cudaStreamSynchronize(stream));
CUDA_CHECK(cudaMemcpy(nodeGains.data(), d_nodeGains, nodesCount * sizeof(float),
          cudaMemcpyDeviceToHost));
}

partitionAssignment = bestAssignment;
if (debug)
cout << "Best cut size achieved in this pass: " << bestCutSize << endl;
return bestCutSize;
}



void iterativeKL(const vector<vector<int>>& nets, vector<int>& partitionAssignment, int nodesCount,
                 int maxPasses, int balanceThreshold,
                 const int* d_netNodes, const int* d_netOffsets, int netsCountDevice,
                 int* d_partitionAssignment, float* d_netCuts,
                 // New node gain device parameters:
                 const int* d_nodeNetList, const int* d_nodeNetOffsets,
                 float* d_nodeGains,
                 cudaStream_t stream, cudaEvent_t event) {
    CUDA_CHECK(cudaMemcpy(d_partitionAssignment, partitionAssignment.data(),
                          nodesCount * sizeof(int), cudaMemcpyHostToDevice));
    float overallBestCut = gpuCalCutSizePersistent(d_netNodes, d_netOffsets, d_partitionAssignment, d_netCuts, netsCountDevice, stream, event);
    vector<int> bestPartition = partitionAssignment;
    
    vector<double> passTimes;
    passTimes.reserve(maxPasses);
    
    for (int pass = 0; pass < maxPasses; pass++) {
        auto startPass = chrono::steady_clock::now();
        if (debug)
            cout << "\n--- Pass " << pass + 1 << " ---\n";
        vector<int> previousPartition = partitionAssignment;
        float passCut = FMAlgorithmPass(nets, partitionAssignment, nodesCount, balanceThreshold,
                                        d_netNodes, d_netOffsets, netsCountDevice,
                                        d_partitionAssignment, d_netCuts,
                                        d_nodeNetList, d_nodeNetOffsets, d_nodeGains,
                                        stream, event);
        auto endPass = chrono::steady_clock::now();
        double durationPass = chrono::duration_cast<chrono::milliseconds>(endPass - startPass).count();
        passTimes.push_back(durationPass);
        
        CUDA_CHECK(cudaMemcpy(d_partitionAssignment, partitionAssignment.data(),
                              nodesCount * sizeof(int), cudaMemcpyHostToDevice));
        float cutSize = gpuCalCutSizePersistent(d_netNodes, d_netOffsets, d_partitionAssignment, d_netCuts, netsCountDevice, stream, event);
        cout << "Cut size after pass " << pass + 1 << ": " << cutSize << endl;
        cout << "Time taken for pass " << pass + 1 << ": " << durationPass << " ms" << endl;
        
        if (debug) {
            cout << "Partition configuration after pass " << pass + 1 << ":\n";
            printPartitionContent(partitionAssignment);
        }
        
        if (cutSize < overallBestCut) {
            overallBestCut = cutSize;
            bestPartition = partitionAssignment;
            if (debug)
                cout << "Pass " << pass + 1 << " improved cut size to " << cutSize << endl;
        } else {
            if (debug)
                cout << "No improvement in pass " << pass + 1 
                     << ". Rolling back to previous partition." << endl;
            partitionAssignment = previousPartition;
            CUDA_CHECK(cudaMemcpy(d_partitionAssignment, partitionAssignment.data(),
                                  nodesCount * sizeof(int), cudaMemcpyHostToDevice));
            float rollbackCutSize = gpuCalCutSizePersistent(d_netNodes, d_netOffsets, d_partitionAssignment, d_netCuts, netsCountDevice, stream, event);
            cout << "Cut size after rollback in pass " << pass + 1 << ": " << rollbackCutSize << endl;
            break;
        }
    }
    
    partitionAssignment = bestPartition;
    cout << "\nFinal Partition after iterative passes:" << endl;
    printPartitionContent(partitionAssignment);
    CUDA_CHECK(cudaMemcpy(d_partitionAssignment, partitionAssignment.data(),
                          nodesCount * sizeof(int), cudaMemcpyHostToDevice));
    float finalCut = gpuCalCutSizePersistent(d_netNodes, d_netOffsets, d_partitionAssignment, d_netCuts, netsCountDevice, stream, event);
    cout << "Final cut size: " << finalCut << endl;
    
    double totalTime = 0.0;
    for (auto t : passTimes)
        totalTime += t;
    double averageTime = (passTimes.empty()) ? 0.0 : totalTime / passTimes.size();
    cout << "Average time for all passes: " << averageTime << " ms" << endl;
}

// ------------------------- Main Function -------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file> [max_passes] [balance_threshold]" << endl;
        return 1;
    }
    string filename = argv[1];
    
    int maxPasses = 5;
    int balanceThreshold = 2;
    if (argc >= 3) {
        maxPasses = atoi(argv[2]);
        if (maxPasses <= 0) {
            cerr << "Invalid max_passes value provided. It must be a positive integer." << endl;
            return 1;
        }
    }
    if (argc >= 4) {
        balanceThreshold = atoi(argv[3]);
        if (balanceThreshold < 0) {
            cerr << "Invalid balance_threshold value provided. It must be a non-negative integer." << endl;
            return 1;
        }
    }
    
    // Set environment variables for OpenMP to reduce idle busy-waiting.
    setenv("OMP_WAIT_POLICY", "passive", 1);
    omp_set_dynamic(0);
    int fixedThreadCount = 16; // Adjust according to your system.
    omp_set_num_threads(fixedThreadCount);
    
    vector<vector<int>> nets;
    vector<int> partition1, partition2;
    int nodesCount = 0;
    readAndPartition(filename, nets, partition1, partition2, nodesCount);

    // fixedThreadCount = 4; // Adjust according to your system.
    // omp_set_num_threads(fixedThreadCount);
    
    vector<int> partitionAssignment(nodesCount, -1);
    for (int node : partition1)
        partitionAssignment[node] = 0;
    for (int node : partition2)
        partitionAssignment[node] = 1;
    
    vector<int> flatNetNodes;
    vector<int> netOffsets;
    flattenNets(nets, flatNetNodes, netOffsets);
    int netsCountDevice = nets.size();
    
    // Build the node-net incidence list (for gain computation)
    vector<int> nodeNetList;
    vector<int> nodeNetOffsets;
    buildNodeIncidenceList(nets, nodesCount, nodeNetList, nodeNetOffsets);
    
    // Allocate device memory for nets.
    int totalNetNodes = flatNetNodes.size();
    int *d_netNodes = nullptr, *d_netOffsets = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_netNodes, totalNetNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_netOffsets, (netsCountDevice + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_netNodes, flatNetNodes.data(), totalNetNodes * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_netOffsets, netOffsets.data(), (netsCountDevice + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    
    // Allocate device memory for partition assignment.
    int* d_partitionAssignment = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_partitionAssignment, nodesCount * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_partitionAssignment, partitionAssignment.data(),
                          nodesCount * sizeof(int), cudaMemcpyHostToDevice));
    
    // Allocate device memory for persistent net cuts.
    float* d_netCuts = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_netCuts, netsCountDevice * sizeof(float)));
    
    // Allocate device memory for node gains.
    float* d_nodeGains = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_nodeGains, nodesCount * sizeof(float)));
    
    // Allocate device memory for flattened node incidence list.
    int* d_nodeNetList = nullptr;
    int* d_nodeNetOffsets = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_nodeNetList, nodeNetList.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_nodeNetOffsets, (nodesCount + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_nodeNetList, nodeNetList.data(), nodeNetList.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodeNetOffsets, nodeNetOffsets.data(), (nodesCount + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    
    // Create one stream and one event to reuse for all GPU cut-size and gain calls.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreate(&event));
    
    float initialGpuCut = gpuCalCutSizePersistent(d_netNodes, d_netOffsets, d_partitionAssignment, d_netCuts, netsCountDevice, stream, event);
    cout << "Initial GPU computed cut size: " << initialGpuCut << endl;
    
    auto startGL = chrono::steady_clock::now();
    GainList* gainList0 = new GainList();
    GainList* gainList1 = new GainList();
    // Create gain lists for each partition using the GPU-computed gains.
    {
        // Launch kernel to compute gains.
        int threadsPerBlock = 256;
        int blocks = (nodesCount + threadsPerBlock - 1) / threadsPerBlock;
        computeNodeGainsKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_netNodes, d_netOffsets,
                                                                       d_nodeNetList, d_nodeNetOffsets,
                                                                       d_partitionAssignment,
                                                                       d_nodeGains,
                                                                       nodesCount);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        vector<float> h_nodeGains(nodesCount);
        CUDA_CHECK(cudaMemcpy(h_nodeGains.data(), d_nodeGains, nodesCount * sizeof(float),
                              cudaMemcpyDeviceToHost));
        // Create gain lists for each partition.
        for (int i = 0; i < nodesCount; i++) {
            FMNode* fmnode = new FMNode(i, h_nodeGains[i]);
            if (partitionAssignment[i] == 0)
                gainList0->insertNode(fmnode);
            else
                gainList1->insertNode(fmnode);
        }
    }
    auto endGL = chrono::steady_clock::now();
    auto durationGL = chrono::duration_cast<chrono::milliseconds>(endGL - startGL).count();
    if (debug)
        cout << "Time to create gain lists: " << durationGL << " ms" << endl;
    
    if (debug) {
        cout << "Gain List for Partition 0:" << endl;
        gainList0->printList();
        cout << "\nGain List for Partition 1:" << endl;
        gainList1->printList();
    }
    delete gainList0;
    delete gainList1;
    
    int numProcs = omp_get_num_procs();
    if (debug)
        cout << "Number of processors available: " << numProcs << endl;
    
    iterativeKL(nets, partitionAssignment, nodesCount, maxPasses, balanceThreshold,
                d_netNodes, d_netOffsets, netsCountDevice, d_partitionAssignment, d_netCuts,
                d_nodeNetList, d_nodeNetOffsets, d_nodeGains, stream, event);
    
    CUDA_CHECK(cudaMemcpy(d_partitionAssignment, partitionAssignment.data(),
                          nodesCount * sizeof(int), cudaMemcpyHostToDevice));
    float finalGpuCut = gpuCalCutSizePersistent(d_netNodes, d_netOffsets, d_partitionAssignment, d_netCuts, netsCountDevice, stream, event);
    cout << "Final GPU computed cut size: " << finalGpuCut << endl;
    
    cudaFree(d_netNodes);
    cudaFree(d_netOffsets);
    cudaFree(d_partitionAssignment);
    cudaFree(d_netCuts);
    cudaFree(d_nodeGains);
    cudaFree(d_nodeNetList);
    cudaFree(d_nodeNetOffsets);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(event));
    
    return 0;
}