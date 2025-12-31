#include "HNSWGraph.h"
#include "DiskVector.h"
#include <cmath>
#include <limits>
#include <queue>
#include <string>
#include <sstream>
#include <iostream>


uint8_t encode(int value)
{
    if (value < 0)
        return static_cast<uint8_t>(value + 128);
    else
        return static_cast<uint8_t>(value + 127);
}

int decode(uint8_t stored_value)
{
    if (stored_value < 128)
        return static_cast<int16_t>(stored_value - 128);
    else
        return static_cast<int16_t>(stored_value - 127);
}

float uniform( 
    float min, 
    float max) 
{
    if (min > max)
    {
        printf("input error\n");
        exit(0);
    }

    float x = min + (max - min) * (float)rand() / (float)RAND_MAX;
    if (x < min || x > max)
    {
        printf("input error\n");
        exit(0);
    }

    return x;
}

float gaussian(  // r.v. from Gaussian(mean, sigma)
    float mu,    // mean (location)
    float sigma) // stanard deviation (scale > 0)
{
    float PI = 3.141592654F;
    float FLOATZERO = 1e-6F;

    if (sigma <= 0.0f)
    {
        printf("input error\n");
        exit(0);
    }

    float u1, u2;
    do
    {
        u1 = uniform(0.0f, 1.0f);
    } while (u1 < FLOATZERO);
    u2 = uniform(0.0f, 1.0f);

    float x = mu + sigma * sqrt(-2.0f * log(u1)) * cos(2.0f * PI * u2);
    return x;
}

namespace lsm_vec
{
using namespace ROCKSDB_NAMESPACE;

    HNSWGraph::HNSWGraph(int m, int mMax, int mLevel, float efConstruction, std::ostream &outFile, int vectorDim, const Config& config)
        : m_(m), m_max_(mMax), m_level_(mLevel), ef_construction_(efConstruction), out_file_(outFile), random_generator_(random_device_()), uniform_distribution_(0, 1), max_layer_(-1), entry_point_(-1)
    {
        if(config.random_seed > 0){
            random_generator_.seed(config.random_seed);
        }

        options_.create_if_missing = true;
        options_.db_paths.emplace_back(rocksdb::DbPath(config.db_path, config.db_target_size));
        options_.statistics = rocksdb::CreateDBStatistics();

        db_ = std::make_unique<rocksdb::RocksGraph>(
            options_,
            EDGE_UPDATE_EAGER,
            ENCODING_TYPE_NONE,
            true
        );

        if (config.vector_storage_type == 1) {
            printf("Using page-based vector storage layout\n");
            vector_storage_ = std::make_unique<PagedVectorStorage>(
                config.vector_file_path,
                static_cast<size_t>(vectorDim),
                config.vec_file_capacity,
                config.paged_max_cached_pages
            );
        } else {
            printf("Using plain vector storage layout\n");
            vector_storage_ = std::make_unique<BasicVectorStorage>(
                config.vector_file_path,
                static_cast<size_t>(vectorDim),
                config.vec_file_capacity
            );
        }
    }
    // Generates a random level for the node
    int HNSWGraph::randomLevel()
    {
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        double r = -log(distribution(random_generator_)) / log(1.0 * m_);

        // std::cout << "r: " << r << std::endl;
        return (int)r;
    }

    // Calculates the Euclidean distance between two points
    float HNSWGraph::euclideanDistance(const std::vector<float> &vectorA, const std::vector<float> &vectorB) const
    {
        float sum = 0.0;
        for (size_t i = 0; i < vectorA.size(); ++i)
        {
            sum += (vectorA[i] - vectorB[i]) * (vectorA[i] - vectorB[i]);
        }
        return std::sqrt(sum);
    }

    void HNSWGraph::insertNode(node_id_t nodeId, const std::vector<float> &vector)
    {
        bool vectorStored = false;  // Track whether we've stored this vector

        int highestLayer = randomLevel();

        if (highestLayer > 0)
        {
            Node newNode{nodeId, vector, {}};
            newNode.neighbors = std::unordered_map<int, std::vector<node_id_t>>();
            nodes_[nodeId] = newNode;
        }

        // ----- First node special case -----
        if (entry_point_ == k_invalid_node_id)
        {
            entry_point_ = nodeId;
            max_layer_   = highestLayer;

            linkNeighborsAsterDB(nodeId, {});

            // For the first node, we can just use nodeId as sectionKey
            node_id_t sectionKey = nodeId;
            vector_storage_->storeVectorToDisk(nodeId, vector, sectionKey);
            vectorStored = true;

            return;
        }

        node_id_t currentEntryPoint = entry_point_;

        // ----- Search down from top layer to (highestLayer+1) to choose entry -----
        for (int l = max_layer_; l > highestLayer; --l)
        {
            std::vector<node_id_t> closest = searchLayer(vector, currentEntryPoint, 1, l);
            if (!closest.empty())
            {
                currentEntryPoint = closest[0];
            }
        }

        // Initial guess for sectionKey: if we have upper layers, use currentEntryPoint,
        // otherwise fall back to global entry_point_.
        node_id_t sectionKey = (max_layer_ >= 1 ? currentEntryPoint : entry_point_);

        // ----- From min(max_layer_, highestLayer) ... down to 0 -----
        for (int l = std::min(max_layer_, highestLayer); l >= 0; --l)
        {
            std::vector<node_id_t> neighbors =
                searchLayer(vector, currentEntryPoint, ef_construction_, l);
            std::vector<node_id_t> selectedNeighbors = 
                selectNeighbors(vector, neighbors, m_, l);
                
            // std::vector<node_id_t> selectedNeighbors;
            // if (l > 0)
            //     selectedNeighbors = selectNeighbors(point, neighbors, M, l);
            // else
            //     selectedNeighbors = selectNeighbors(point, neighbors, Mmax, l);

            // If we are at level 1, refine sectionKey using closest neighbor at l=1.
            if (l == 1)
            {
                if (!neighbors.empty())
                    sectionKey = neighbors[0];
                else
                    sectionKey = currentEntryPoint;
            }

            // When we first reach level 0, store the vector using sectionKey
            if (l == 0 && !vectorStored)
            {
                vector_storage_->storeVectorToDisk(nodeId, vector, sectionKey);
                vectorStored = true;
            }

            // Link neighbors as before
            if (l > 0)
            {
                linkNeighbors(nodeId, selectedNeighbors, l);
            }
            else // l == 0
            {
                linkNeighborsAsterDB(nodeId, selectedNeighbors);
            }

            // ---- Shrink connections ----
            if (l > 0)
            {
                for (int neighbor : selectedNeighbors)
                {
                    std::vector<node_id_t> eConn = nodes_[neighbor].neighbors[l];
                    if (eConn.size() > static_cast<size_t>(m_max_))
                    {
                        std::vector<node_id_t> eNewConn =
                            selectNeighbors(nodes_[neighbor].point, eConn, m_max_, l);
                        nodes_[neighbor].neighbors[l] = std::move(eNewConn);
                    }
                }
            }
            else // l == 0
            {
                for (node_id_t neighbor : selectedNeighbors)
                {
                    rocksdb::Edges edges;
                    db_->GetAllEdges(neighbor, &edges);

                    if (edges.num_edges_out > static_cast<uint32_t>(m_max_))
                    {
                        std::vector<node_id_t> eConns;
                        eConns.reserve(edges.num_edges_out);
                        for (uint32_t i = 0; i < edges.num_edges_out; ++i)
                        {
                            eConns.push_back(edges.nxts_out[i].nxt);
                        }

                        std::vector<float> neighborVector;
                        vector_storage_->readVectorFromDisk(neighbor, neighborVector);

                        std::vector<node_id_t> eNewConn =
                            selectNeighbors(neighborVector, eConns, m_max_, l);

                        for (auto node : eConns)
                        {
                            if (std::find(eNewConn.begin(), eNewConn.end(), node) == eNewConn.end())
                            {
                                db_->DeleteEdge(neighbor, node);
                                db_->DeleteEdge(node, neighbor);
                            }
                        }
                    }
                }
            }

            if (!neighbors.empty())
            {
                currentEntryPoint = neighbors[0];
            }
        }

        if (highestLayer > max_layer_)
        {
            entry_point_ = nodeId;
            max_layer_   = highestLayer;
            std::cout << "Updated entry point to node " << nodeId
                    << " at layer " << highestLayer << std::endl;
        }

        // Safety: in principle vectorStored must be true if we reached here.
        if (!vectorStored)
        {
            vector_storage_->storeVectorToDisk(nodeId, vector, sectionKey);
        }

    }

    // Links neighbors for upper layers stored in memory
    void HNSWGraph::linkNeighbors(node_id_t nodeId, const std::vector<node_id_t> &neighborIds, int layer)
    {
        for (node_id_t neighborId : neighborIds)
        {
            nodes_[neighborId].neighbors[layer].push_back(nodeId);
            nodes_[nodeId].neighbors[layer].push_back(neighborId);
        }
    }

    void HNSWGraph::linkNeighborsAsterDB(node_id_t nodeId, const std::vector<node_id_t> &neighborIds)
    {
        db_->AddVertex(nodeId);

        for (node_id_t neighborId : neighborIds)
        {
            db_->AddEdge(nodeId, neighborId);
            db_->AddEdge(neighborId, nodeId);
        }
    }

    std::vector<node_id_t> HNSWGraph::selectNeighbors(
        const std::vector<float> &vector,
        const std::vector<node_id_t> &candidateIds,
        int maxNeighbors,
        int layer)
    {
        if (!use_heuristic_neighbor_selection_) {
            return selectNeighborsSimple(vector, candidateIds, maxNeighbors, layer);
        } else {
            return selectNeighborsHeuristic2(vector, candidateIds, maxNeighbors, layer);
        }
    }

    // Selects neighbors based on distance and pruning logic
    std::vector<node_id_t> HNSWGraph::selectNeighborsSimple(const std::vector<float> &vector, const std::vector<node_id_t> &candidateIds, int maxNeighbors, int layer)
    {
        if (candidateIds.size() <= static_cast<size_t>(maxNeighbors))
        {
            return candidateIds;
        }
        else
        {
            std::priority_queue<std::pair<float, node_id_t>> topCandidates;
            for (node_id_t candidateId : candidateIds)
            {
                float dist = 0.0;
                if (layer > 0)
                {
                    dist = euclideanDistance(vector, nodes_[candidateId].point);
                }
                else if (layer == 0)
                {
                    std::vector<float> candidateVector;
                    vector_storage_->readVectorFromDisk(candidateId, candidateVector);
                    dist = euclideanDistance(vector, candidateVector);
                }

                topCandidates.emplace(dist, candidateId);
                if (topCandidates.size() > static_cast<size_t>(maxNeighbors))
                {
                    topCandidates.pop();
                }
            }

            std::vector<node_id_t> selectedNeighbors;
            while (!topCandidates.empty())
            {
                selectedNeighbors.push_back(topCandidates.top().second);
                topCandidates.pop();
            }
            return selectedNeighbors;
        }
    }

    std::vector<node_id_t> HNSWGraph::selectNeighborsHeuristic1(
        const std::vector<float> &vector,
        const std::vector<node_id_t> &candidateIds,
        int maxNeighbors,
        int layer)
    {
        // If there are not enough candidates, behave like original code.
        if (candidateIds.size() <= static_cast<size_t>(maxNeighbors)) {
            return candidateIds;
        }

        struct CandidateInfo {
            node_id_t   id;
            float distToQuery;
        };

        std::vector<CandidateInfo> candInfos;
        candInfos.reserve(candidateIds.size());

        // Optional cache for layer 0 to avoid reading the same vectors multiple times.
        // For layer > 0 we can use nodes_[id].point directly.
        std::unordered_map<node_id_t, std::vector<float>> vecCache;

        auto getVector = [&](node_id_t nodeId) -> const std::vector<float>& {
            if (layer > 0) {
                return nodes_[nodeId].point;
            } else {
                auto it = vecCache.find(nodeId);
                if (it != vecCache.end()) return it->second;

                std::vector<float> v;
                vector_storage_->readVectorFromDisk(nodeId, v);

                auto res = vecCache.emplace(nodeId, std::move(v));
                return res.first->second;
            }
        };

        // 1) Precompute distances query -> candidate
        for (node_id_t candidateId : candidateIds)
        {
            const auto& candVec = getVector(candidateId);
            float d = euclideanDistance(vector, candVec);
            candInfos.push_back(CandidateInfo{candidateId, d});
        }

        // 2) Sort by distance to query (ascending)
        std::sort(
            candInfos.begin(),
            candInfos.end(),
            [](const CandidateInfo& a, const CandidateInfo& b) {
                return a.distToQuery < b.distToQuery;
            }
        );

        // 3) Heuristic (diversified neighbor selection):
        std::vector<node_id_t> selected;
        selected.reserve(maxNeighbors);
        std::vector<node_id_t> rejected;

        for (const auto& cand : candInfos)
        {
            if (selected.size() >= static_cast<size_t>(maxNeighbors))
                break;

            node_id_t candidateId = cand.id;
            float distToQuery = cand.distToQuery;

            bool good = true;
            for (node_id_t selectedId : selected)
            {
                const auto& v1 = getVector(selectedId);
                const auto& v2 = getVector(candidateId);
                float currentDist = euclideanDistance(v1, v2);

                if (currentDist < distToQuery)
                {
                    good = false;
                    break;
                }
            }

            if (good)
                selected.push_back(candidateId);
            else
                rejected.push_back(candidateId);
        }

        // 4) If we still have fewer than maxNeighbors neighbors, fill from rejected,
        //    preserving the order by distToQuery (because 'rejected' was
        //    filled while scanning candInfos in sorted order).
        for (node_id_t rejectedId : rejected)
        {
            if (selected.size() >= static_cast<size_t>(maxNeighbors))
                break;
            selected.push_back(rejectedId);
        }

        return selected;
    }

    std::vector<node_id_t> HNSWGraph::selectNeighborsHeuristic2(
        const std::vector<float> &vector,
        const std::vector<node_id_t> &candidateIds,
        int maxNeighbors,
        int layer)
    {
        // If there are not enough candidates, behave like original code.
        if (candidateIds.size() <= static_cast<size_t>(maxNeighbors)) {
            return candidateIds;
        }

        struct CandidateInfo {
            node_id_t   id;
            float distToQuery;
        };

        std::vector<CandidateInfo> candInfos;
        candInfos.reserve(candidateIds.size());

        // Optional cache for layer 0 to avoid reading the same vectors multiple times.
        // For layer > 0 we can use nodes_[id].point directly.
        std::unordered_map<node_id_t, std::vector<float>> vecCache;

        auto getVector = [&](node_id_t nodeId) -> const std::vector<float>& {
            if (layer > 0) {
                return nodes_[nodeId].point;
            } else {
                auto it = vecCache.find(nodeId);
                if (it != vecCache.end()) return it->second;

                std::vector<float> v;
                vector_storage_->readVectorFromDisk(nodeId, v);

                auto res = vecCache.emplace(nodeId, std::move(v));
                return res.first->second;
            }
        };

        // 1) Precompute distances query -> candidate
        for (node_id_t candidateId : candidateIds)
        {
            const auto& candVec = getVector(candidateId);
            float d = euclideanDistance(vector, candVec);
            candInfos.push_back(CandidateInfo{candidateId, d});
        }

        // 2) Sort by distance to query (ascending)
        std::sort(
            candInfos.begin(),
            candInfos.end(),
            [](const CandidateInfo& a, const CandidateInfo& b) {
                return a.distToQuery < b.distToQuery;
            }
        );

        // 3) Heuristic (diversified neighbor selection):
        std::vector<node_id_t> selected;
        selected.reserve(maxNeighbors);
        std::vector<node_id_t> rejected;

        for (const auto& cand : candInfos)
        {
            if (selected.size() >= static_cast<size_t>(maxNeighbors))
                break;

            node_id_t candidateId = cand.id;
            float distToQuery = cand.distToQuery;

            bool good = true;
            for (node_id_t selectedId : selected)
            {
                const auto& v1 = getVector(selectedId);
                const auto& v2 = getVector(candidateId);
                float currentDist = euclideanDistance(v1, v2);

                if (currentDist < distToQuery)
                {
                    good = false;
                    break;
                }
            }

            if (good)
                selected.push_back(candidateId);
            else
                rejected.push_back(candidateId);
        }

        return selected;
    }

    std::vector<node_id_t> HNSWGraph::searchLayer(const std::vector<float>& queryVector,
                                             node_id_t entryPointId,
                                             int efSearch,
                                             int layer)
    {
        if (layer < 0) {
            std::cerr << "Error: Invalid layer for search.\n";
            return {};
        }
        if (efSearch <= 0) {
            return {};
        }

        // Visited set
        std::unordered_set<node_id_t> visited;
        visited.reserve(static_cast<std::size_t>(efSearch) * 4);

        // Candidates: max-heap by (-distance) => smallest distance comes first
        using Cand = std::pair<float, node_id_t>;
        std::priority_queue<Cand> candidates;

        // W: max-heap by (distance) => farthest among current best is on top
        std::priority_queue<Cand> nearest;

        auto getDistance = [&](node_id_t nodeId) -> float {
            if (layer > 0) {
                // Upper layers: vectors are in-memory
                return euclideanDistance(queryVector, nodes_.at(nodeId).point);
            } else {
                // Level 0: vectors are on disk
                std::vector<float> v;
                vector_storage_->readVectorFromDisk(nodeId, v);
                return euclideanDistance(queryVector, v);
            }
        };

        // Initialize with entry point
        float distToEntry = getDistance(entryPointId);
        visited.insert(entryPointId);
        candidates.emplace(-distToEntry, entryPointId);
        nearest.emplace(distToEntry, entryPointId);

        while (!candidates.empty()) {
            node_id_t currentId = candidates.top().second;
            float currentDist = -candidates.top().first;
            candidates.pop();

            // Early termination: if closest candidate is worse than the worst in W, stop
            if (!nearest.empty() && currentDist > nearest.top().first) {
                break;
            }

            if (layer > 0) {
                // Upper layers: adjacency is in memory.
                const auto nodeIt = nodes_.find(currentId);
                if (nodeIt == nodes_.end()) {
                    continue; // Defensive: should not happen
                }

                const auto& neighborMap = nodeIt->second.neighbors;
                auto it = neighborMap.find(layer);
                if (it == neighborMap.end()) {
                    continue; // No adjacency list at this layer (do not create it)
                }

                const auto& neighborIds = it->second;
                for (node_id_t neighborId : neighborIds) {
                    if (visited.insert(neighborId).second) {
                        float d = euclideanDistance(queryVector, nodes_.at(neighborId).point);
                        if (static_cast<int>(nearest.size()) < efSearch || d < nearest.top().first) {
                            candidates.emplace(-d, neighborId);
                            nearest.emplace(d, neighborId);
                            if (static_cast<int>(nearest.size()) > efSearch) {
                                nearest.pop();
                            }
                        }
                    }
                }
            } else {
                // Level 0: adjacency is stored in RocksGraph.
                rocksdb::Edges edges;
                db_->GetAllEdges(currentId, &edges);

                for (uint32_t i = 0; i < edges.num_edges_out; ++i) {
                    node_id_t neighborId = static_cast<node_id_t>(edges.nxts_out[i].nxt);

                    if (visited.insert(neighborId).second) {
                        std::vector<float> neighborVec;
                        vector_storage_->readVectorFromDisk(neighborId, neighborVec);

                        float d = euclideanDistance(queryVector, neighborVec);
                        if (static_cast<int>(nearest.size()) < efSearch || d < nearest.top().first) {
                            candidates.emplace(-d, neighborId);
                            nearest.emplace(d, neighborId);
                            if (static_cast<int>(nearest.size()) > efSearch) {
                                nearest.pop();
                            }
                        }
                    }
                }
            }
        }

        // Extract results from nearest (sorted ascending by distance)
        std::vector<std::pair<float, node_id_t>> temp;
        temp.reserve(nearest.size());
        while (!nearest.empty()) {
            temp.push_back(nearest.top());
            nearest.pop();
        }

        std::sort(temp.begin(), temp.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<node_id_t> result;
        result.reserve(temp.size());
        for (const auto& p : temp) {
            result.push_back(p.second);
        }
        return result;
    }


    // std::vector<node_id_t> HNSWGraph::searchLayer(const std::vector<float> &queryPoint, node_id_t entryPoint, int ef, int layer)
    // {
    //     // set of visited elements
    //     std::unordered_set<node_id_t> visited;

    //     // set of candidates
    //     std::priority_queue<std::pair<float, node_id_t>> candidates; // C: set of candidates

    //     // dynamic list of found nearest neighbors
    //     std::priority_queue<std::pair<float, node_id_t>> nearestNeighbors; // W: dynamic list of found nearest neighbors

    //     if (layer > 0)
    //     {
    //         // initialize the search
    //         float distToEP = euclideanDistance(queryPoint, nodes[entryPoint].point);
    //         visited.insert(entryPoint); // v ← ep
    //         candidates.emplace(-distToEP, entryPoint);
    //         nearestNeighbors.emplace(distToEP, entryPoint);

    //         while (!candidates.empty())
    //         {
    //             // get the nearest candidate, extract nearest element from C
    //             node_id_t current = candidates.top().second;
    //             float currentDist = -candidates.top().first;
    //             candidates.pop();

    //             // check if the current candidate is closer than the farthest neighbor
    //             // get furthest element from W to q
    //             if (currentDist > nearestNeighbors.top().first)
    //             {
    //                 break;
    //             }
    //             for (node_id_t neighbor : nodes[current].neighbors[layer])
    //             {
    //                 if (visited.find(neighbor) == visited.end())
    //                 {
    //                     visited.insert(neighbor);
    //                     float dist = euclideanDistance(queryPoint, nodes[neighbor].point);
    //                     if (nearestNeighbors.size() < ef || dist < nearestNeighbors.top().first)
    //                     {
    //                         candidates.emplace(-dist, neighbor);
    //                         nearestNeighbors.emplace(dist, neighbor);
    //                         if (nearestNeighbors.size() > ef)
    //                         {
    //                             nearestNeighbors.pop();
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         std::vector<std::pair<float, node_id_t>> temp;

            
    //         while (!nearestNeighbors.empty())
    //         {
    //             temp.push_back(nearestNeighbors.top());
    //             nearestNeighbors.pop();
    //         }

    //         std::sort(temp.begin(), temp.end(), [](const std::pair<float, node_id_t> &a, const std::pair<float, node_id_t> &b)
    //                   {
    //                       return a.first < b.first;
    //                   });

    //         std::vector<node_id_t> result;
    //         for (const auto &pair : temp)
    //         {
    //             result.push_back(pair.second);
    //         }
    //         return result;
    //     }
    //     else if (layer == 0)
    //     {
    //         // initialize the search
    //         // float distToEP = euclideanDistance(queryPoint, nodes[entryPoint].point);
    //         std::vector<float> entryPointVector;

    //         vectorStorage->readVectorFromDisk(entryPoint, entryPointVector);

    //         float distToEP = euclideanDistance(queryPoint, entryPointVector);
    //         visited.insert(entryPoint); // v ← ep
    //         candidates.emplace(-distToEP, entryPoint);
    //         nearestNeighbors.emplace(distToEP, entryPoint);

    //         while (!candidates.empty())
    //         {
    //             // extract nearest element from C to q
    //             // get furthest element from W to q
    //             node_id_t current = candidates.top().second;
    //             float currentDist = -candidates.top().first;
    //             candidates.pop();

    //             // check if the current candidate is closer than the farthest neighbor
    //             // get furthest element from W to q
    //             if (currentDist > nearestNeighbors.top().first)
    //             {
    //                 break;
    //             }

    //             rocksdb::Edges edges;
    //             db_->GetAllEdges(current, &edges);

    //             std::vector<int> neighbors;
    //             for (uint32_t i = 0; i < edges.num_edges_out; ++i)
    //             {
    //                 neighbors.push_back(edges.nxts_out[i].nxt);
    //             }

    //             for (node_id_t neighbor : neighbors)
    //             {
    //                 if (visited.find(neighbor) == visited.end())
    //                 {
    //                     visited.insert(neighbor);
    //                     // float dist = euclideanDistance(queryPoint, nodes[neighbor].point);
    //                     std::vector<float> neighborVector;
    //                     vectorStorage->readVectorFromDisk(neighbor, neighborVector);

    //                     float dist = euclideanDistance(queryPoint, neighborVector);
    //                     if (nearestNeighbors.size() < ef || dist < nearestNeighbors.top().first)
    //                     {
    //                         candidates.emplace(-dist, neighbor);
    //                         nearestNeighbors.emplace(dist, neighbor);
    //                         if (nearestNeighbors.size() > ef)
    //                         {
    //                             nearestNeighbors.pop(); // remove furthest element from W to q
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         std::vector<std::pair<float, node_id_t>> temp;

    //         while (!nearestNeighbors.empty())
    //         {
    //             temp.push_back(nearestNeighbors.top());
    //             nearestNeighbors.pop();
    //         }

    //         std::sort(temp.begin(), temp.end(), [](const std::pair<float, node_id_t> &a, const std::pair<float, node_id_t> &b)
    //                   {
    //                       return a.first < b.first; // 比较距离
    //                   });

    //         std::vector<node_id_t> result;
    //         for (const auto &pair : temp)
    //         {
    //             result.push_back(pair.second);
    //         }
    //         return result;
    //     }
    //     else
    //     {
    //         // error
    //         std::cerr << "Error: Invalid layer for search." << std::endl;
    //     }
    // }

    // Performs a greedy search to find the closest neighbor at a specific layer
    node_id_t HNSWGraph::knnSearch(const std::vector<float> &queryVector)
    {
        // W ← ∅ set for the current nearest elements
        std::vector<node_id_t> nearestNeighbors; // W: dynamic list of found nearest neighbors

        // ep ← get enter point for hnsw
        node_id_t currentEntryPoint = entry_point_;
        int currentLayer = max_layer_;

        for (int l = max_layer_; l >= 1; --l)
        {
            std::vector<node_id_t> nearestNeighbors = searchLayer(queryVector, currentEntryPoint, 30, l);
            currentEntryPoint = nearestNeighbors[0];
        }
        nearestNeighbors = searchLayer(queryVector, currentEntryPoint, 30, 0);

        return nearestNeighbors[0];
    }

    void HNSWGraph::printState() const
    {
        // We do not print layer 0 by request.
        if (max_layer_ <= 0) {
            std::cout << "HNSW state: max_layer=" << max_layer_
                    << " (no upper layers to report)\n";
            return;
        }

        // Count how many nodes have adjacency info at each upper layer.
        // Note: this counts "nodes that currently have neighbor entries at layer l",
        // which is derivable from existing in-memory structures without extra metadata.
        std::vector<std::size_t> layerNodeCounts(static_cast<std::size_t>(max_layer_ + 1), 0);

        // To avoid double counting, we track per-layer seen node IDs.
        std::vector<std::unordered_set<node_id_t>> seen(static_cast<std::size_t>(max_layer_ + 1));

        for (const auto& kv : nodes_) {
            node_id_t nodeId = kv.first;
            const Node& node = kv.second;

            for (const auto& kv2 : node.neighbors) {
                // The key type should be "layer index".
                // If your neighbors map key is int, this is already int.
                // If it's not int, cast safely.
                int layer = static_cast<int>(kv2.first);

                if (layer <= 0) continue;          // skip layer 0 (and any invalid)
                if (layer > max_layer_) continue;    // defensive

                // Option A (default): count node if it has the layer key at all.
                // Option B: count only if the adjacency list is non-empty:
                // if (kv2.second.empty()) continue;

                if (seen[static_cast<std::size_t>(layer)].insert(nodeId).second) {
                    layerNodeCounts[static_cast<std::size_t>(layer)]++;
                }
            }
        }

        std::cout << "HNSW state:\n";
        std::cout << "  max_layer: " << max_layer_ << "\n";
        std::cout << "  upper_layers: [1.." << max_layer_ << "]\n";

        // Print top-down for readability
        for (int l = max_layer_; l >= 1; --l) {
            std::cout << "  layer " << l << ": "
                    << layerNodeCounts[static_cast<std::size_t>(l)]
                    << " nodes\n";
        }
    }

    // void HNSWGraph::printStatistics() const
    // {
    //     std::cout << "Indexing Time: " << indexingTime << " seconds" << std::endl;

    //     std::cout << "-------graph part------" << std::endl;
    //     std::cout << "Total Aster I/O Time: " << ioTime << " seconds" << std::endl;
    //     std::cout << "Read Operations: " << readIOCount << ", Time: " << readIOTime << " seconds" << std::endl;
    //     std::cout << "Write Node Operations: " << writenodeIOCount << ", Time: " << writenodeIOTime << " seconds" << std::endl;
    //     std::cout << "Add Edge Operations: " << addedgeIOCount << ", Time: " << addedgeIOTime << " seconds" << std::endl;
    //     std::cout << "Delete Edge Operations: " << deleteedgeIOCount << ", Time: " << deleteedgeIOTime << " seconds" << std::endl;
    //     std::cout << "-------vector part------" << std::endl;
    //     std::cout << "Total Vector I/O Time: " << vecreadtime + vecwritetime << " seconds" << std::endl;
    //     std::cout << "Vector Read Operations: " << vecreadcount << ", Time: " << vecreadtime << " seconds" << std::endl;
    //     std::cout << "Vector Write Operations: " << vecwritecount << ", Time: " << vecwritetime << " seconds" << std::endl;
    //     std::cout << std::endl;
    //     std::cout << std::endl;
    // }
} // namespace ROCKSDB_NAMESPACE
