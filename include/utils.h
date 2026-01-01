#ifndef HNSW_UTILS_H
#define HNSW_UTILS_H

#include <vector>
#include <string>
#include "lsm_vec_index.h" // Include your HNSWGraph class definition

// Reads vectors from a bvecs file
std::vector<std::vector<float>> readBvecsFile(const std::string& filename);

// Reads ground truth indices from an ivecs file
std::vector<std::vector<int>> readIvecsFile(const std::string& filename);

// Inserts vectors from a bvecs file into the HNSW graph
void insertFromBigANNFile(lsm_vec::LSMVec& graph, const std::string& filename);

// Queries the HNSW graph and compares results with ground truth
void queryBigANN(lsm_vec::LSMVec& graph, const std::string& queryFile, const std::string& groundTruthFile);

std::vector<std::vector<float>> readFvecsFile(const std::string& filename);

void insertFromFile(lsm_vec::LSMVec& graph, const std::string& filename);

void queryAndCompareWithGroundTruth(lsm_vec::LSMVec& graph, const std::string& queryFile, const std::string& groundTruthFile);

int getdim(const std::string& filename);
#endif // HNSW_UTILS_H