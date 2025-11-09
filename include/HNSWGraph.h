#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include "rocksdb/graph.h"
#include "rocksdb/db.h"
#include <iostream>
#include "DiskVector.h"
#include "SimHashSampler.h"
#include "GraphReorderer.h"

namespace ROCKSDB_NAMESPACE
{
    class HNSWGraph
    {
    public:
        struct Node
        {
            int id;
            std::vector<float> point;
            std::unordered_map<int, std::vector<int>> neighbors;
        };

        HNSWGraph(int M, int Mmax, int Ml, float efConstruction, RocksGraph *db, 
                  std::ostream &outFile, std::string vectorfilePath, int dim,
                  bool enableSampling = true, bool enableReordering = true);

        void insertNode(int id, const std::vector<float> &point);
        int KNNsearch(const std::vector<float> &queryPoint);
        void printGraphStructure();
        std::unordered_set<int> highestLayerNodes;
        void printStatistics() const;

        int ioCount;
        double ioTime;
        double indexingTime;
        size_t readIOCount = 0;
        size_t writenodeIOCount = 0;
        size_t addedgeIOCount = 0;
        size_t deleteedgeIOCount = 0;
        double readIOTime = 0.0;
        double writenodeIOTime = 0.0;
        double addedgeIOTime = 0.0;
        double deleteedgeIOTime = 0.0;
        size_t readVertexPropertyCount = 0;
        size_t readEdgesCount = 0;
        double readVertexPropertyTime = 0.0;
        double readEdgesTime = 0.0;
        
        std::string vectorfilePath;
        int vectordim = 0;
        std::unique_ptr<VectorStorage> vectorStorage;
        void debugPrintGraph();

        size_t vecreadcount = 0;
        double vecreadtime = 0.0;
        size_t vecwritecount = 0;
        double vecwritetime = 0.0;

        bool enableSampling;
        bool enableReordering;
        std::unique_ptr<SimHashSampler> sampler;
        std::unique_ptr<GraphReorderer> reorderer;
        
        float samplingRatio = 0.8f;
        float epsilon = 0.1f;
        int hashBits = 32;
        int reorderWindowSize = 64;
        float reorderLambda = 1.0f;
        int reorderInterval = 10000;
        int insertionsSinceReorder = 0;
        
        void maybeReorder();
        void reorderGraph();

    private:
        int randomLevel();
        float euclideanDistance(const std::vector<float> &a, const std::vector<float> &b) const;
        std::vector<int> searchLayer(const std::vector<float> &queryPoint, int ep, int ef, int layer);
        std::vector<int> searchLayerWithSampling(const std::vector<float> &queryPoint, 
                                                 int ep, int ef, int layer,
                                                 const std::vector<int8_t> *queryHash);
        void linkNeighbors(int nodeId, const std::vector<int> &neighbors, int layer);
        void linkNeighborsAsterDB(int nodeId, const std::vector<float> &point, const std::vector<int> &neighbors);
        std::vector<int> selectNeighbors(const std::vector<float> &point, const std::vector<int> &candidates, int M, int layer);
        
        int M;
        int Mmax;
        int Ml;
        float efConstruction;
        RocksGraph *db;
        std::ostream &outFile;
        int entryPoint = -1;
        std::unordered_map<int, Node> nodes;
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<> dist;
        int maxLayer;
        std::vector<float> node_length;
    };
} // namespace ROCKSDB_NAMESPACE