#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <limits>
#include <chrono>
#include <random>
#include <memory>
#include <sys/stat.h>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <iomanip>

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_num_procs() 1
#endif

using namespace std;

bool EIG_init = false;
string EIG_file;
string fout_name;

struct sparseMatrix {
    unsigned int nodeNum;
    vector<unordered_map<int, float>> adjacencyList;
    vector<int> split[2];
    vector<int> remain[2];
    vector<float> nodeGains;
    unordered_map<int, vector<int>> nodeConnections;
    
    explicit sparseMatrix(unsigned int size) : nodeNum(size) {
        adjacencyList.resize(size);
        nodeGains.resize(size, 0.0f);
    }
    
    void initNodeConnections() {
        #pragma omp parallel for schedule(dynamic)
        for (unsigned int i = 0; i < nodeNum; i++) {
            vector<int> connected;
            connected.reserve(adjacencyList[i].size() * 2);
            for (const auto& [j, _] : adjacencyList[i])
                connected.push_back(j);
            for (unsigned int j = 0; j < i; j++) {
                if (adjacencyList[j].find(i) != adjacencyList[j].end())
                    connected.push_back(j);
            }
            #pragma omp critical
            nodeConnections[i] = std::move(connected);
        }
    }
};

float getEdgeWeight(const sparseMatrix& spMat, int node1, int node2) {
    if (node1 > node2)
        swap(node1, node2);
    const auto& neighbors = spMat.adjacencyList[node1];
    auto it = neighbors.find(node2);
    return (it != neighbors.end()) ? it->second : 0.0f;
}

void InitializeSparsMatrix(const string& filename, sparseMatrix& spMat) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    
    string line;
    getline(fin, line);
    long int netsNum, nodesNum;
    stringstream(line) >> netsNum >> nodesNum;
    
    spMat = sparseMatrix(nodesNum);
    vector<int> nodes;
    nodes.reserve(1000);
    long int nonZeroElements = 0;
    
    for (int i = 0; i < netsNum; i++) {
        getline(fin, line);
        stringstream ss(line);
        nodes.clear();
        int node;
        while (ss >> node)
            nodes.push_back(node - 1); // Convert to 0-based indexing
        float weight = 1.0f / (nodes.size() - 1);
        for (size_t j = 0; j < nodes.size(); j++) {
            for (size_t k = j + 1; k < nodes.size(); k++) {
                int node1 = nodes[j], node2 = nodes[k];
                if (node1 > node2)
                    swap(node1, node2);
                spMat.adjacencyList[node1][node2] += weight;
                nonZeroElements++;
            }
        }
    }
    fin.close();
}

string getBaseName(const string& path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == string::npos) ? path : path.substr(pos + 1);
}

void createDir(const string& dirName) {
    struct stat info;
    if (stat(dirName.c_str(), &info) != 0) {
        #ifdef _WIN32
            _mkdir(dirName.c_str());
        #else
            mkdir(dirName.c_str(), 0755);
        #endif
    }
}

void shuffleSparceMatrix(sparseMatrix& spMat) {
    for (auto& p : spMat.split)
        p.clear();
    for (auto& r : spMat.remain)
        r.clear();
    
    if (EIG_init) {
        ifstream fEIG(EIG_file);
        string line;
        getline(fEIG, line);
        getline(fEIG, line);
        while (getline(fEIG, line)) {
            int node, split_side;
            double weight;
            stringstream(line) >> node >> split_side >> weight;
            spMat.split[split_side].push_back(node);
            spMat.remain[split_side].push_back(node);
        }
    } else {
        vector<int> nodes(spMat.nodeNum);
        iota(nodes.begin(), nodes.end(), 0);
        shuffle(nodes.begin(), nodes.end(), mt19937(random_device{}()));
        size_t mid = spMat.nodeNum / 2;
        spMat.split[0].assign(nodes.begin(), nodes.begin() + mid);
        spMat.remain[0] = spMat.split[0];
        spMat.split[1].assign(nodes.begin() + mid, nodes.end());
        spMat.remain[1] = spMat.split[1];
    }
}

float calCutSize(const sparseMatrix& spMat) {
    float cutSize = 0.0f;
    unordered_set<int> right(spMat.remain[1].begin(), spMat.remain[1].end());
    #pragma omp parallel for reduction(+:cutSize)
    for (size_t i = 0; i < spMat.remain[0].size(); i++) {
        int node = spMat.remain[0][i];
        for (const auto& [nbr, w] : spMat.adjacencyList[node])
            if (right.count(nbr))
                cutSize += w;
        for (int nbr : right) {
            if (nbr < node) {
                auto it = spMat.adjacencyList[nbr].find(node);
                if (it != spMat.adjacencyList[nbr].end())
                    cutSize += it->second;
            }
        }
    }
    return cutSize;
}

float connections(const sparseMatrix& spMat, int node) {
    float ext = 0.0f, in = 0.0f;
    unordered_set<int> left(spMat.split[0].begin(), spMat.split[0].end());
    for (const auto& [nbr, w] : spMat.adjacencyList[node]) {
        if (left.count(nbr))
            in += w;
        else
            ext += w;
    }
    for (int i = 0; i < node; i++) {
        auto it = spMat.adjacencyList[i].find(node);
        if (it != spMat.adjacencyList[i].end()) {
            if (left.count(i))
                in += it->second;
            else
                ext += it->second;
        }
    }
    return ext - in;
}

void updateAffectedNodeGains(sparseMatrix& spMat, int node1, int node2) {
    vector<int> affected(spMat.nodeConnections[node1]);
    affected.insert(affected.end(), spMat.nodeConnections[node2].begin(), spMat.nodeConnections[node2].end());
    sort(affected.begin(), affected.end());
    affected.erase(unique(affected.begin(), affected.end()), affected.end());
    #pragma omp parallel for
    for (size_t i = 0; i < affected.size(); i++) {
        spMat.nodeGains[affected[i]] = connections(spMat, affected[i]);
    }
}

void swip(sparseMatrix& spMat, int num1, int num2) {
    spMat.remain[0].erase(find(spMat.remain[0].begin(), spMat.remain[0].end(), num1));
    spMat.remain[1].erase(find(spMat.remain[1].begin(), spMat.remain[1].end(), num2));
    *find(spMat.split[0].begin(), spMat.split[0].end(), num1) = num2;
    *find(spMat.split[1].begin(), spMat.split[1].end(), num2) = num1;
}

void KL(sparseMatrix& spMat) {
    shuffleSparceMatrix(spMat);
    spMat.initNodeConnections();
    ofstream fout(fout_name);
    int iteration = 0, terminate = 0, limit = log2(spMat.nodeNum) + 5;
    float cutSize = calCutSize(spMat);
    float initialCut = cutSize; // Save initial cut size
    fout << "0\t" << cutSize << "\t0" << endl;

    #pragma omp parallel for
    for (size_t i = 0; i < spMat.nodeNum; i++)
        spMat.nodeGains[i] = connections(spMat, i);

    while (!spMat.remain[0].empty() && !spMat.remain[1].empty()) {
        float maxGain = -numeric_limits<float>::max(), minGain = numeric_limits<float>::max();
        int maxIdx = -1, minIdx = -1;
        for (size_t i = 0; i < spMat.remain[0].size(); i++) {
            int n = spMat.remain[0][i];
            if (spMat.nodeGains[n] > maxGain) {
                maxGain = spMat.nodeGains[n];
                maxIdx = i;
            }
        }
        for (size_t i = 0; i < spMat.remain[1].size(); i++) {
            int n = spMat.remain[1][i];
            if (spMat.nodeGains[n] < minGain) {
                minGain = spMat.nodeGains[n];
                minIdx = i;
            }
        }
        if (maxIdx >= 0 && minIdx >= 0) {
            int n1 = spMat.remain[0][maxIdx];
            int n2 = spMat.remain[1][minIdx];
            float gain = maxGain - minGain - 2 * getEdgeWeight(spMat, n1, n2);
            cutSize -= gain;
            swip(spMat, n1, n2);
            updateAffectedNodeGains(spMat, n1, n2);
            iteration++;
            fout << iteration << "\t" << cutSize << "\t" << gain << endl;
            cout << "Iteration " << iteration << ": Cut Size = " << cutSize << endl;
            if (gain <= 0 && ++terminate > limit)
                break;
            else if (gain > 0)
                terminate = 0;
        } else {
            break;
        }
    }
    fout.close();
    cout << "Initial Cut Size: " << initialCut << endl;
    cout << "Final Cut Size: " << cutSize << endl;
}

int main(int argc, char *argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    createDir("results");
    createDir("pre_saved_EIG");

    if (argc != 2 && argc != 3) {
        cout << "Usage: " << argv[0] << " <input_file> [-EIG]" << endl;
        return 1;
    }

    string input_file = argv[1];
    string base_name = getBaseName(input_file);
    fout_name = "results/" + base_name + "_KL_CutSize_output.txt";

    if (argc == 3 && strcmp(argv[2], "-EIG") == 0) {
        EIG_init = true;
        EIG_file = "pre_saved_EIG/" + base_name + "_out.txt";
        fout_name = "results/" + base_name + "_KL_CutSize_EIG_output.txt";
    }

    // Start runtime measurement.
    auto start = chrono::steady_clock::now();
    
    sparseMatrix spMat(1);
    InitializeSparsMatrix(input_file, spMat);
    #ifdef _OPENMP
        omp_set_num_threads(omp_get_num_procs());
    #endif
    
    KL(spMat);
    
    auto end = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Execution Time: " << elapsed << " ms" << endl;
    return 0;
}