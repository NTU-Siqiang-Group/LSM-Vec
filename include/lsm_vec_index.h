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
#include "disk_vector.h"
#include "config.h"
#include "statistics.h"
#include "logger.h"

namespace lsm_vec
{
using namespace ROCKSDB_NAMESPACE;

    class LSMVec
    {
    public:
        struct Node
        {
            node_id_t id;
            std::vector<float> point;
            std::unordered_map<int, std::vector<node_id_t>> neighbors; // Layer -> neighbors
        };

        HNSWStats stats;

        bool use_heuristic_neighbor_selection_ = true;

        void setNeighborSelection(bool useHeuristic) {
            use_heuristic_neighbor_selection_ = useHeuristic;
        }

        LSMVec(int m, int mMax, int mLevel, float efConstruction, std::ostream &outFile, int vectorDim, const Config& config);

        void insertNode(node_id_t nodeId, const std::vector<float> &vector);
        node_id_t knnSearch(const std::vector<float> &queryVector);
        std::unordered_set<int> highest_layer_nodes_;

        void printIndexStatus() const;
        void printStatistics() const;
        void printState() const;

        std::string vector_file_path_;
        int vector_dim_ = 0;
        std::unique_ptr<IVectorStorage> vector_storage_;

    private:
        int randomLevel();
        float euclideanDistance(const std::vector<float> &vectorA, const std::vector<float> &vectorB) const;
        std::vector<node_id_t> searchLayer(const std::vector<float> &queryVector, node_id_t entryPointId, int efSearch, int layer);
        void linkNeighbors(node_id_t nodeId, const std::vector<node_id_t> &neighborIds, int layer);
        void linkNeighborsAsterDB(node_id_t nodeId, const std::vector<node_id_t> &neighborIds);

        std::vector<node_id_t> selectNeighbors(
            const std::vector<float>& vector,
            const std::vector<node_id_t>& candidateIds,
            int maxNeighbors,
            int layer
        );

        std::vector<node_id_t> selectNeighborsSimple(
            const std::vector<float>& vector,
            const std::vector<node_id_t>& candidateIds,
            int maxNeighbors,
            int layer
        );

        std::vector<node_id_t> selectNeighborsHeuristic1(
            const std::vector<float>& vector,
            const std::vector<node_id_t>& candidateIds,
            int maxNeighbors,
            int layer
        );

        std::vector<node_id_t> selectNeighborsHeuristic2(
            const std::vector<float>& vector,
            const std::vector<node_id_t>& candidateIds,
            int maxNeighbors,
            int layer
        );

        int m_;                      // Number of established connections
        int m_max_;                  // Maximum neighbors per layer
        int m_level_;                // Normalization factor for level generation
        float ef_construction_;      // Parameter for candidate selection

        rocksdb::Options options_;
        std::unique_ptr<rocksdb::RocksGraph> db_;
        std::ostream &out_file_;     // Output stream for logging

        std::unordered_map<int, Node> nodes_; // In-memory nodes for layers > 0

        std::random_device random_device_;
        std::mt19937 random_generator_;
        std::uniform_real_distribution<> uniform_distribution_;
        int max_layer_;
        node_id_t entry_point_ = k_invalid_node_id; // Entry point for HNSW graph

        // Store length for node
        std::vector<float> node_lengths_;
    };
} // namespace ROCKSDB_NAMESPACE
