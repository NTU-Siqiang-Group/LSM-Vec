#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include "rocksdb/graph.h"
#include "rocksdb/db.h"
#include <iostream>
#include "DiskVector.h"
#include "Config.h"
#include "Statistics.h"

namespace ROCKSDB_NAMESPACE
{
    class HNSWGraph
    {
    public:
        struct Node
        {
            int id;
            std::vector<float> point;
            std::unordered_map<int, std::vector<int>> neighbors; // Layer -> neighbors
        };

        HNSWStats stats;

        bool useHeuristicNeighborSelection_ = true;

        void setNeighborSelection(bool v) {
            useHeuristicNeighborSelection_ = v;
        }

        HNSWGraph(int M, int Mmax, int Ml, float efConstruction, RocksGraph *db, std::ostream &outFile, std::string vectorfilePath, int dim, const Config& cfg);

        void insertNode(int id, const std::vector<float> &point);
        void insertNodeOld(int id, const std::vector<float> &point);
        int KNNsearch(const std::vector<float> &queryPoint);
        std::unordered_set<int> highestLayerNodes;

        void printIndexStatus() const;
        void printStatistics() const;

        std::string vectorfilePath;
        int vectordim = 0;
        std::unique_ptr<IVectorStorage> vectorStorage;

    private:
        int randomLevel();
        float euclideanDistance(const std::vector<float> &a, const std::vector<float> &b) const;
        std::vector<int> searchLayer(const std::vector<float> &queryPoint, int ep, int ef, int layer);
        void linkNeighbors(int nodeId, const std::vector<int> &neighbors, int layer);
        void linkNeighborsAsterDB(int nodeId, const std::vector<float> &point, const std::vector<int> &neighbors);

        std::vector<int> selectNeighbors(
            const std::vector<float>& point,
            const std::vector<int>& candidates,
            int M,
            int layer
        );

        std::vector<int> selectNeighborsSimple(
            const std::vector<float>& point,
            const std::vector<int>& candidates,
            int M,
            int layer
        );

        std::vector<int> selectNeighborsHeuristic1(
            const std::vector<float>& point,
            const std::vector<int>& candidates,
            int M,
            int layer
        );

        std::vector<int> selectNeighborsHeuristic2(
            const std::vector<float>& point,
            const std::vector<int>& candidates,
            int M,
            int layer
        );

        int M;                // number of established connections
        int Mmax;             // Maximum neighbors per layer
        int Ml;               // Nomalization factor for level generation
        float efConstruction; // Parameter for candidate selection

        RocksGraph *db;        // Pointer to RocksGraph (AsterDB)
        std::ostream &outFile; // Output stream for logging

        int entryPoint = -1; // Entry point for HNSW graph

        std::unordered_map<int, Node> nodes; // In-memory nodes for layers > 0

        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<> dist;
        int maxLayer;

        // Store length for node
        std::vector<float> node_length;
    };
} // namespace ROCKSDB_NAMESPACE
#pragma once