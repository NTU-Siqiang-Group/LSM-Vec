#include "HNSWUtils.h"
#include <fstream>
#include <iostream>

// Function to read vectors from a bvecs file
std::vector<std::vector<float>> readBvecsFile(const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<std::vector<float>> data;

    if (!input.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    while (input)
    {
        int dim;
        input.read(reinterpret_cast<char *>(&dim), sizeof(int));
        if (!input)
            break;

        std::vector<uint8_t> vec(dim);
        input.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(uint8_t));

        if (input)
        {
            // Convert uint8_t to float for compatibility with HNSW
            std::vector<float> floatVec(vec.begin(), vec.end());
            data.push_back(floatVec);
        }
    }

    input.close();
    return data;
}

// Function to read ground truth from an ivecs file
std::vector<std::vector<int>> readIvecsFile(const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<std::vector<int>> data;

    if (!input.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    while (input)
    {
        int k;
        input.read(reinterpret_cast<char *>(&k), sizeof(int));
        if (!input)
            break;

        std::vector<int> vec(k);
        input.read(reinterpret_cast<char *>(vec.data()), k * sizeof(int));
        if (input)
        {
            data.push_back(vec);
        }
    }

    input.close();
    return data;
}

// Function to insert vectors from a bvecs file into the HNSW graph
void insertFromBigANNFile(ROCKSDB_NAMESPACE::HNSWGraph &graph, const std::string &filename)
{
    auto data = readBvecsFile(filename);
    for (size_t i = 0; i < data.size(); ++i)
    {
        graph.insertNode(static_cast<int>(i), data[i]);
    }
}

// Function to perform queries from a file and compare results with ground truth
void queryBigANN(ROCKSDB_NAMESPACE::HNSWGraph &graph, const std::string &queryFile, const std::string &groundTruthFile)
{
    auto queries = readBvecsFile(queryFile);
    auto groundTruth = readIvecsFile(groundTruthFile);

    if (queries.size() != groundTruth.size())
    {
        std::cerr << "Error: The number of queries and ground truth entries do not match." << std::endl;
        return;
    }

    int correctMatches = 0;
    int totalQueries = static_cast<int>(queries.size());

    for (size_t i = 0; i < queries.size(); ++i)
    {
        int hnswResult = graph.KNNsearch(queries[i]);
        int groundTruthResult = groundTruth[i][0]; // Assuming the first entry is the closest

        if (hnswResult == groundTruthResult)
        {
            ++correctMatches;
        }

        std::cout << "Query " << i << ": HNSW Nearest Neighbor = " << hnswResult
                  << ", Ground Truth = " << groundTruthResult << std::endl;
    }

    // Calculate and print accuracy
    float accuracy = static_cast<float>(correctMatches) / totalQueries;
    std::cout << "Accuracy: " << (accuracy * 100) << "%" << std::endl;
}

std::vector<std::vector<float>> readFvecsFile(const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<std::vector<float>> data;

    if (!input.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    while (input)
    {
        int dim;
        input.read(reinterpret_cast<char *>(&dim), sizeof(int));
        if (!input)
            break;

        std::vector<float> vec(dim);
        input.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
        if (input)
        {
            data.push_back(vec);
        }
    }

    input.close();
    return data;
}

int getdim(const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    int dim = 0;

    if (!input.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return dim;
    }

    while (input)
    {
        input.read(reinterpret_cast<char *>(&dim), sizeof(int));
        break;
    }

    input.close();
    return dim;
}

// Function to insert vectors from an fvecs file into the HNSW graph
void insertFromFile(ROCKSDB_NAMESPACE::HNSWGraph &graph, const std::string &filename)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    int dim = 0;
    size_t node_count = 0;

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();

    bool isFvecs = filename.find(".fvecs") != std::string::npos;
    bool isBvecs = filename.find(".bvecs") != std::string::npos;

    if (!isFvecs && !isBvecs)
    {
        std::cerr << "Error: Unsupported file format. Only fvecs and bvecs files are supported." << std::endl;
        return;
    }

    while (input.read(reinterpret_cast<char *>(&dim), sizeof(int)) && node_count <= 100000000)
    {
        if (node_count == 0)
        {
            graph.vectordim = dim;
            std::cout << "vector dim: " << graph.vectordim << std::endl;
        }

        std::vector<float> floatVec;
        std::vector<float> finalVec(dim);

        if (isFvecs)
        {
            floatVec.resize(dim);
            input.read(reinterpret_cast<char *>(floatVec.data()), dim * sizeof(float));
            finalVec = floatVec; // fvecs直接是float，不需要转换
        }
        else if (isBvecs)
        {
            std::vector<uint8_t> byteVec(dim);
            input.read(reinterpret_cast<char *>(byteVec.data()), dim * sizeof(uint8_t));

            // 归一化到0~1（或其他标准化策略，取决于你的HNSW实现需求）
            for (int j = 0; j < dim; ++j)
            {
                finalVec[j] = static_cast<float>(byteVec[j]); // 不归一化直接放
            }
        }

        if (node_count == 99999000)
        {
            start = std::chrono::high_resolution_clock::now(); // 只对后面1000个点计时
        }

        graph.insertNode(static_cast<int>(node_count), finalVec);

        if (node_count % 100000 == 0)
        {
            std::cout << "Inserting node " << node_count << std::endl;
        }

        ++node_count;
    }

    end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    std::cout << "Inserting 1000 nodes took " << duration << " seconds" << std::endl;

    input.close();
}

// Function to perform queries from a file and compare results with ground truth
void queryAndCompareWithGroundTruth(ROCKSDB_NAMESPACE::HNSWGraph &graph, const std::string &queryFile, const std::string &groundTruthFile)
{
    auto queries = readFvecsFile(queryFile);
    auto groundTruth = readIvecsFile(groundTruthFile);

    if (queries.size() != groundTruth.size())
    {
        std::cerr << "Error: The number of queries and ground truth entries do not match." << std::endl;
        return;
    }

    int correctMatches = 0;
    int totalQueries = static_cast<int>(queries.size());
    double totalQueryTime = 0.0;

    for (size_t i = 0; i < queries.size(); ++i)
    {
        // Start measuring query time
        auto start = std::chrono::high_resolution_clock::now();

        // Perform the HNSW query
        int hnswResult = graph.KNNsearch(queries[i]);

        // Stop measuring query time
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> queryTime = end - start;

        // Accumulate query time
        totalQueryTime += queryTime.count();

        // Get ground truth result
        int groundTruthResult = groundTruth[i][0]; // Assuming the first entry is the closest

        // Check if the result matches the ground truth
        if (hnswResult == groundTruthResult)
        {
            ++correctMatches;
        }
    }

    // Calculate and print accuracy
    float accuracy = static_cast<float>(correctMatches) / totalQueries;
    std::cout << "Accuracy: " << (accuracy * 100) << "%" << std::endl;

    // Print efficiency statistics
    std::cout << "Total Query Time: " << totalQueryTime << " ms" << std::endl;
    std::cout << "Average Query Time: " << (totalQueryTime / totalQueries) << " ms/query" << std::endl;
}
