#ifndef HNSW_UTILS_H
#define HNSW_UTILS_H

#include <vector>
#include <string>
#include "HNSWGraph.h"

// Reads vectors from a bvecs file
std::vector<std::vector<float>> readBvecsFile(const std::string& filename);

// Reads ground truth indices from an ivecs file
std::vector<std::vector<int>> readIvecsFile(const std::string& filename);

// Inserts vectors from a bvecs file into the HNSW graph
void insertFromBigANNFile(ROCKSDB_NAMESPACE::HNSWGraph& graph, const std::string& filename);

// Queries the HNSW graph and compares results with ground truth
void queryBigANN(ROCKSDB_NAMESPACE::HNSWGraph& graph, const std::string& queryFile, const std::string& groundTruthFile);

std::vector<std::vector<float>> readFvecsFile(const std::string& filename);

void insertFromFile(ROCKSDB_NAMESPACE::HNSWGraph& graph, const std::string& filename);

void queryAndCompareWithGroundTruth(ROCKSDB_NAMESPACE::HNSWGraph& graph, const std::string& queryFile, const std::string& groundTruthFile);

int getdim(const std::string& filename);
#endif // HNSW_UTILS_H
