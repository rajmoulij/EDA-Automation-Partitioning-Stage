#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <string>
#include <chrono>

#ifdef _OPENMP
  #include <omp.h>
#else
  // Dummy definitions when OpenMP is disabled.
  inline int omp_get_num_procs() { return 1; }
  inline void omp_set_num_threads(int) { }
#endif

using namespace std;

// Global debug flag: Set to true to enable debug prints.
bool debug = false;

// ------------------------------------------------------------------------------
// FM Node structure representing a node with a gain value.
struct FMNode {
    int id;         // Node identifier
    float gain;     // Gain value
    FMNode* next;   // Pointer to the next node in the doubly linked list
    FMNode* prev;   // Pointer to the previous node

    FMNode(int id, float gain) : id(id), gain(gain), next(nullptr), prev(nullptr) {}
};

// ------------------------------------------------------------------------------
// Doubly linked list tailored for FM algorithm gains.
// Nodes are maintained in descending order by gain.
// ------------------------------------------------------------------------------
class GainList {
private:
    FMNode* head;
    FMNode* tail;
public:
    GainList() : head(nullptr), tail(nullptr) {}
    ~GainList() { clear(); }
    
    // Delete all nodes.
    void clear() {
        while (head) {
            FMNode* temp = head;
            head = head->next;
            delete temp;
        }
        tail = nullptr;
    }

    // Insert a node in descending order by gain.
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
        if (current == nullptr) { // Insert at the tail.
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

    // For debugging: Print the nodes in the gain list.
    void printList() const {
        FMNode* current = head;
        while (current) {
            cout << "Node ID: " << current->id << ", Gain: " << current->gain << endl;
            current = current->next;
        }
    }
};

// ------------------------------------------------------------------------------
// Helper function: printPartitionContent
// Prints the current partition assignment content using node IDs (0-based).
// ------------------------------------------------------------------------------
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

// ------------------------------------------------------------------------------
// Helper function: calCutSize
// Computes the final cut size given the nets and the current partition assignment.
// Each net contributes a cost if it is "cut" (has nodes in both partitions).
// ------------------------------------------------------------------------------
float calCutSize(const vector<vector<int>>& nets, const vector<int>& partitionAssignment) {
    float totalCost = 0.0f;
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:totalCost)
#endif
    for (size_t netIdx = 0; netIdx < nets.size(); netIdx++) {
        const auto &net = nets[netIdx];
        if (net.size() < 2)
            continue;
        float weight = 1.0f / (net.size() - 1);
        for (size_t i = 0; i < net.size(); i++) {
            for (size_t j = i + 1; j < net.size(); j++) {
                if (partitionAssignment[net[i]] != partitionAssignment[net[j]])
                    totalCost += weight;
            }
        }
    }
    return totalCost;
}

// ------------------------------------------------------------------------------
// Helper functions to read the input file and partition nodes.
// ------------------------------------------------------------------------------
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
    // Read the header line containing netsCount and nodesCount.
    getline(infile, line);
    {
        stringstream ss(line);
        ss >> netsCount >> nodesCount;
    }

    // Resize nets in advance.
    nets.resize(netsCount);

    // Read all net lines sequentially.
    vector<string> netLines(netsCount);
    for (int i = 0; i < netsCount; i++) {
        getline(infile, netLines[i]);
    }
    infile.close();

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < netsCount; i++) {
        stringstream netStream(netLines[i]);
        int node;
        vector<int> netNodes;
        while (netStream >> node) {
            // Convert to 0-based indexing.
            netNodes.push_back(node - 1);
        }
        nets[i] = move(netNodes);
    }

    // Create the list of nodes.
    vector<int> nodes(nodesCount);
    for (int i = 0; i < nodesCount; i++) {
        nodes[i] = i;
    }

    // Shuffle nodes (std::shuffle is not parallelized).
    random_device rd;
    mt19937 gen(rd());
    shuffle(nodes.begin(), nodes.end(), gen);

    // Partition the nodes into two groups.
    int mid = nodesCount / 2;
    partition1.assign(nodes.begin(), nodes.begin() + mid);
    partition2.assign(nodes.begin() + mid, nodes.end());
}

// ------------------------------------------------------------------------------
// Function: computeGainForNode
// For each net that contains the node, counts connections to nodes in the same partition ("internal")
// and nodes in the other partition ("external"). Gain is defined as: gain = external - internal.
// ------------------------------------------------------------------------------
float computeGainForNode(int node, const vector<vector<int>>& nets, const vector<int>& partitionAssignment) {
    float internal = 0, external = 0;
    int myPart = partitionAssignment[node];
    for (const auto &net : nets) {
        bool found = false;
        for (int n : net) {
            if (n == node) { found = true; break; }
        }
        if (!found) continue;
        for (int n : net) {
            if (n == node)
                continue;
            if (partitionAssignment[n] == myPart)
                internal++;
            else
                external++;
        }
    }
    return external - internal;
}

// ------------------------------------------------------------------------------
// Function: createGainListForPartition
// Creates a gain list for a given partition using OpenMP if available.
// ------------------------------------------------------------------------------
GainList* createGainListForPartition(const vector<int>& partition,
                                     const vector<vector<int>>& nets,
                                     const vector<int>& partitionAssignment) {
    GainList* gl = new GainList();
    int pSize = partition.size();

#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < pSize; i++) {
            int node = partition[i];
            float gain = computeGainForNode(node, nets, partitionAssignment);
            FMNode* fmnode = new FMNode(node, gain);
            #pragma omp critical
            {
                gl->insertNode(fmnode);
            }
        }
    }
#else
    for (int i = 0; i < pSize; i++) {
        int node = partition[i];
        float gain = computeGainForNode(node, nets, partitionAssignment);
        FMNode* fmnode = new FMNode(node, gain);
        gl->insertNode(fmnode);
    }
#endif
    return gl;
}

// ------------------------------------------------------------------------------
// Function: FMAlgorithmPass
// Performs one pass of the FM/KL algorithm while recording the partition assignment
// that minimizes the actual cut cost. Rollback is done to that state.
// ------------------------------------------------------------------------------
float FMAlgorithmPass(const vector<vector<int>>& nets, vector<int>& partitionAssignment, int nodesCount, int balanceThreshold) {
    vector<bool> locked(nodesCount, false);

    // Record the initial cut cost and partition assignment.
    float initialCutSize = calCutSize(nets, partitionAssignment);
    float bestCutSize = initialCutSize;
    vector<int> bestAssignment = partitionAssignment;

    // Precompute initial gains for all nodes.
    vector<float> nodeGains(nodesCount, 0.0f);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < nodesCount; i++) {
        nodeGains[i] = computeGainForNode(i, nets, partitionAssignment);
    }
    
    int moves = 0;
    while (moves < nodesCount) {
        // Compute number of nodes in each partition.
        int count0 = 0, count1 = 0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:count0, count1)
#endif
        for (int i = 0; i < nodesCount; i++) {
            if (partitionAssignment[i] == 0)
                count0++;
            else
                count1++;
        }
        int diff = count0 - count1; // Positive if Partition 0 is larger.
        
        // Find the unlocked node with maximum gain while enforcing balance.
        float maxGain = -1e9;
        int bestNode = -1;
        for (int i = 0; i < nodesCount; i++) {
            if (!locked[i]) {
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
        
        // Move bestNode to the opposite partition.
        partitionAssignment[bestNode] = 1 - partitionAssignment[bestNode];
        locked[bestNode] = true;
        moves++;
        
        // Compute current cut size after the move.
        float currentCutSize = calCutSize(nets, partitionAssignment);
        if (debug)
            cout << "Move " << moves << " - Moved node " << bestNode 
                 << ", Current cut size: " << currentCutSize << endl;
        // Record the partition assignment if an improvement is observed.
        if (currentCutSize < bestCutSize) {
            bestCutSize = currentCutSize;
            bestAssignment = partitionAssignment;
            if (debug)
                cout << "New best cut size: " << bestCutSize << endl;
        }
        
        // Update gains for nodes sharing a net with bestNode.
        for (const auto &net : nets) {
            bool containsBest = false;
            for (int n : net) {
                if (n == bestNode) { containsBest = true; break; }
            }
            if (!containsBest)
                continue;
            for (int n : net) {
                if (!locked[n] && n != bestNode) {
                    nodeGains[n] = computeGainForNode(n, nets, partitionAssignment);
                }
            }
        }
    }
    // Roll back to the partition assignment with the lowest cut size observed in this pass.
    partitionAssignment = bestAssignment;
    if (debug)
        cout << "Best cut size achieved in this pass: " << bestCutSize << endl;
    return bestCutSize;
}

// ------------------------------------------------------------------------------
// Function: iterativeKL
// Performs several passes of the FM/KL algorithm.
// ------------------------------------------------------------------------------
void iterativeKL(const vector<vector<int>>& nets, vector<int>& partitionAssignment, int nodesCount, int maxPasses, int balanceThreshold) {
    // Use the initial cut cost as the overall best.
    float overallBestCut = calCutSize(nets, partitionAssignment);
    vector<int> bestPartition = partitionAssignment;
    
    // Vector to store time durations (in milliseconds) for each pass.
    vector<double> passTimes;
    passTimes.reserve(maxPasses);
    
    for (int pass = 0; pass < maxPasses; pass++) {
        // Start timing this pass.
        auto startPass = chrono::steady_clock::now();
        
        if (debug)
            cout << "\n--- Pass " << pass + 1 << " ---\n";
        
        // Save current partition assignment as backup.
        vector<int> previousPartition = partitionAssignment;
        float passCut = FMAlgorithmPass(nets, partitionAssignment, nodesCount, balanceThreshold);
        
        // End timing this pass.
        auto endPass = chrono::steady_clock::now();
        double durationPass = chrono::duration_cast<chrono::milliseconds>(endPass - startPass).count();
        passTimes.push_back(durationPass);
        
        // Compute and print the final cut size after this pass.
        float cutSize = calCutSize(nets, partitionAssignment);
        cout << "Cut size after pass " << pass + 1 << ": " << cutSize << endl;
        cout << "Time taken for pass " << pass + 1 << ": " << durationPass << " ms" << endl;
        
        // If debug is enabled, also print the partition configuration.
        if (debug) {
            cout << "Partition configuration after pass " << pass + 1 << ":\n";
            printPartitionContent(partitionAssignment);
        }
        
        // Update overall best if improved.
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
            float rollbackCutSize = calCutSize(nets, partitionAssignment);
            cout << "Cut size after rollback in pass " << pass + 1 << ": " << rollbackCutSize << endl;
            break;
        }
    }
    
    // Update final partition assignment.
    partitionAssignment = bestPartition;
    cout << "\nFinal Partition after iterative passes:" << endl;
    printPartitionContent(partitionAssignment);
    float finalCut = calCutSize(nets, partitionAssignment);
    cout << "Final cut size: " << finalCut << endl;
    
    // Compute and print the average time for all passes.
    double totalTime = 0.0;
    for (auto t : passTimes)
        totalTime += t;
    double averageTime = (passTimes.empty()) ? 0.0 : totalTime / passTimes.size();
    cout << "Average time for all passes: " << averageTime << " ms" << endl;
}

// ------------------------------------------------------------------------------
// Main function: Reads input, partitions nodes, sets the number of threads using the "-t" flag,
// and runs the iterative KL pass.
// ------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0]
             << " <input_file> [max_passes] [balance_threshold] [-t num_threads]" << endl;
        return 1;
    }

    // The first argument is required: input file.
    string filename = argv[1];

    // Set default values.
    int maxPasses = 5;
    int balanceThreshold = 2;
    int numThreads = 0;  // A value of 0 means no override (use OpenMP default)

    // Process the remaining arguments.
    // Positional arguments (without flag) are assumed to be maxPasses and balanceThreshold.
    // The "-t" flag specifies the number of threads.
    int positionalCount = 0;
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-t") {
            if (i + 1 < argc) {
                numThreads = atoi(argv[i + 1]);
                i++;  // Skip the next argument.
            } else {
                cerr << "Missing thread count after -t flag." << endl;
                return 1;
            }
        } else {
            // Process positional arguments.
            if (positionalCount == 0) {
                maxPasses = atoi(argv[i]);
                if (maxPasses <= 0) {
                    cerr << "Invalid max_passes value. It must be a positive integer." << endl;
                    return 1;
                }
            } else if (positionalCount == 1) {
                balanceThreshold = atoi(argv[i]);
                if (balanceThreshold < 0) {
                    cerr << "Invalid balance_threshold value. It must be non-negative." << endl;
                    return 1;
                }
            }
            positionalCount++;
        }
    }

#ifdef _OPENMP
    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
        cout << "Number of OpenMP threads set to: " << numThreads << endl;
    }
#endif

    vector<vector<int>> nets;
    vector<int> partition1, partition2;
    int nodesCount = 0;
    
    readAndPartition(filename, nets, partition1, partition2, nodesCount);
    
    // Build initial partition assignment: nodes from partition1 get 0, nodes from partition2 get 1.
    vector<int> partitionAssignment(nodesCount, -1);
    for (int node : partition1)
        partitionAssignment[node] = 0;
    for (int node : partition2)
        partitionAssignment[node] = 1;
    
    // Print initial cut size.
    float initialCut = calCutSize(nets, partitionAssignment);
    cout << "Initial cut size: " << initialCut << endl;
    
    // Create gain lists (if debug enabled, print timing details).
    auto startGL = chrono::steady_clock::now();
    GainList* gainList0 = createGainListForPartition(partition1, nets, partitionAssignment);
    GainList* gainList1 = createGainListForPartition(partition2, nets, partitionAssignment);
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
    
    iterativeKL(nets, partitionAssignment, nodesCount, maxPasses, balanceThreshold);
    
    return 0;
}