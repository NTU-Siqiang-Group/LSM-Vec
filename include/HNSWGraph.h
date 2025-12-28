#pragma once
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include "rocksdb/graph.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/statistics.h"
#include <iostream>
#include "DiskVector.h"
#include "Config.h"
#include "Statistics.h"

namespace lsm_vec
{
using namespace ROCKSDB_NAMESPACE;

    class HNSWGraph
    {
    public:
        struct Node
        {
            node_id_t id;
            std::vector<float> point;
            std::unordered_map<node_id_t, std::vector<node_id_t>> neighbors; // Layer -> neighbors
        };

        HNSWStats stats;

        bool useHeuristicNeighborSelection_ = true;

        void setNeighborSelection(bool v) {
            useHeuristicNeighborSelection_ = v;
        }

        HNSWGraph(int M, int Mmax, int Ml, float efConstruction, std::ostream &outFile, std::string vectorfilePath, int dim, const Config& cfg_);

        void insertNode(node_id_t id, const std::vector<float> &point);
        void insertNodeOld(node_id_t id, const std::vector<float> &point);
        node_id_t KNNsearch(const std::vector<float> &queryPoint);
        std::unordered_set<int> highestLayerNodes;

        void printIndexStatus() const;
        void printStatistics() const;

        std::string vectorfilePath;
        int vectordim = 0;
        std::unique_ptr<IVectorStorage> vectorStorage;

    private:
        int randomLevel();
        float euclideanDistance(const std::vector<float> &a, const std::vector<float> &b) const;
        std::vector<node_id_t> searchLayer(const std::vector<float> &queryPoint, node_id_t entryPoint, int ef, int layer);
        void linkNeighbors(node_id_t id, const std::vector<node_id_t> &neighbors, int layer);
        void linkNeighborsAsterDB(node_id_t id, const std::vector<float> &point, const std::vector<node_id_t> &neighbors);

        std::vector<node_id_t> selectNeighbors(
            const std::vector<float>& point,
            const std::vector<node_id_t>& candidates,
            int M,
            int layer
        );

        std::vector<node_id_t> selectNeighborsSimple(
            const std::vector<float>& point,
            const std::vector<node_id_t>& candidates,
            int M,
            int layer
        );

        std::vector<node_id_t> selectNeighborsHeuristic1(
            const std::vector<float>& point,
            const std::vector<node_id_t>& candidates,
            int M,
            int layer
        );

        std::vector<node_id_t> selectNeighborsHeuristic2(
            const std::vector<float>& point,
            const std::vector<node_id_t>& candidates,
            int M,
            int layer
        );

        int M;                // number of established connections
        int Mmax;             // Maximum neighbors per layer
        int Ml;               // Nomalization factor for level generation
        float efConstruction; // Parameter for candidate selection

        rocksdb::Options options_;
        std::unique_ptr<rocksdb::RocksGraph> db_;
        std::ostream &outFile; // Output stream for logging

        std::unordered_map<int, Node> nodes; // In-memory nodes for layers > 0

        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<> dist;
        int maxLayer;
        node_id_t entryPoint = -1; // Entry point for HNSW graph

        // Store length for node
        std::vector<float> node_length;
    };
} // namespace ROCKSDB_NAMESPACE
