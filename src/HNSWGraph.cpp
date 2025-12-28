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

    HNSWGraph::HNSWGraph(int M, int Mmax, int Ml, float efConstruction, std::ostream &outFile, int vectordim, const Config& cfg_)
        : M(M), Mmax(Mmax), Ml(Ml), efConstruction(efConstruction), outFile(outFile), gen(rd()), dist(0, 1), maxLayer(-1), entryPoint(-1)
    {
        if(cfg_.random_seed > 0){
            gen.seed(cfg_.random_seed);
        }

        options_.create_if_missing = true;
        options_.db_paths.emplace_back(rocksdb::DbPath(cfg_.db_path, cfg_.db_target_size));
        options_.statistics = rocksdb::CreateDBStatistics();

        db_ = std::make_unique<rocksdb::RocksGraph>(
            options_,
            EDGE_UPDATE_EAGER,
            ENCODING_TYPE_NONE,
            true
        );
        //vectorStorage = std::make_unique<VectorStorage>(vectorfilePath, vectordim, 100000000);
        //vectorStorage = std::make_unique<VectorStorage>(vectorfilePath, vectordim);
        if (cfg_.vector_storage_type == 1) {
            printf("Using page-based vector storage layout\n");
            vectorStorage = std::make_unique<PagedVectorStorage>(
                cfg_.vector_file_path,
                static_cast<size_t>(vectordim),
                cfg_.vec_file_capacity,
                cfg_.paged_max_cached_pages
            );
        } else {
            printf("Using plain vector storage layout\n");
            vectorStorage = std::make_unique<BasicVectorStorage>(
                cfg_.vector_file_path,
                static_cast<size_t>(vectordim),
                cfg_.vec_file_capacity
            );
        }
    }
    // Generates a random level for the node
    int HNSWGraph::randomLevel()
    {
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        double r = -log(distribution(gen)) / log(1.0 * M);

        // std::cout << "r: " << r << std::endl;
        return (int)r;
    }

    // Calculates the Euclidean distance between two points
    float HNSWGraph::euclideanDistance(const std::vector<float> &a, const std::vector<float> &b) const
    {
        float sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(sum);
    }

    void HNSWGraph::insertNode(node_id_t id, const std::vector<float> &point)
    {
        bool vectorStored = false;  // NEW: track whether we've stored this vector

        int highestLayer = randomLevel();

        if (highestLayer > 0)
        {
            Node newNode{id, point, {}};
            newNode.neighbors = std::unordered_map<int, std::vector<node_id_t>>();
            nodes[id] = newNode;
        }

        // ----- First node special case -----
        if (entryPoint == k_invalid_node_id)
        {
            entryPoint = id;
            maxLayer   = highestLayer;

            linkNeighborsAsterDB(id, {});

            // For the first node, we can just use id as sectionKey
            node_id_t sectionKey = id;
            vectorStorage->storeVectorToDisk(id, point, sectionKey);
            vectorStored = true;

            return;
        }

        node_id_t currentEntryPoint = entryPoint;

        // ----- Search down from top layer to (highestLayer+1) to choose entry -----
        for (int l = maxLayer; l > highestLayer; --l)
        {
            std::vector<node_id_t> closest = searchLayer(point, currentEntryPoint, 1, l);
            if (!closest.empty())
            {
                currentEntryPoint = closest[0];
            }
        }

        // Initial guess for sectionKey: if we have upper layers, use currentEntryPoint,
        // otherwise fall back to global entryPoint.
        node_id_t sectionKey = (maxLayer >= 1 ? currentEntryPoint : entryPoint);

        // ----- From min(maxLayer, highestLayer) ... down to 0 -----
        for (int l = std::min(maxLayer, highestLayer); l >= 0; --l)
        {
            std::vector<node_id_t> neighbors =
                searchLayer(point, currentEntryPoint, efConstruction, l);
            std::vector<node_id_t> selectedNeighbors = 
                selectNeighbors(point, neighbors, M, l);
                
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

            // *** NEW: when we first reach level 0, store the vector using sectionKey ***
            if (l == 0 && !vectorStored)
            {
                vectorStorage->storeVectorToDisk(id, point, sectionKey);
                vectorStored = true;
            }

            // Link neighbors as before
            if (l > 0)
            {
                linkNeighbors(id, selectedNeighbors, l);
            }
            else // l == 0
            {
                linkNeighborsAsterDB(id, selectedNeighbors);
            }

            // ---- Shrink connections ----
            if (l > 0)
            {
                for (int neighbor : selectedNeighbors)
                {
                    std::vector<node_id_t> eConn = nodes[neighbor].neighbors[l];
                    if (eConn.size() > static_cast<size_t>(Mmax))
                    {
                        std::vector<node_id_t> eNewConn =
                            selectNeighbors(nodes[neighbor].point, eConn, Mmax, l);
                        nodes[neighbor].neighbors[l] = std::move(eNewConn);
                    }
                }
            }
            else // l == 0
            {
                for (node_id_t neighbor : selectedNeighbors)
                {
                    rocksdb::Edges edges;
                    db_->GetAllEdges(neighbor, &edges);

                    if (edges.num_edges_out > static_cast<uint32_t>(Mmax))
                    {
                        std::vector<node_id_t> eConns;
                        eConns.reserve(edges.num_edges_out);
                        for (uint32_t i = 0; i < edges.num_edges_out; ++i)
                        {
                            eConns.push_back(edges.nxts_out[i].nxt);
                        }

                        std::vector<float> cur_point2;
                        vectorStorage->readVectorFromDisk(neighbor, cur_point2);

                        std::vector<node_id_t> eNewConn =
                            selectNeighbors(cur_point2, eConns, Mmax, l);

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

        if (highestLayer > maxLayer)
        {
            entryPoint = id;
            maxLayer   = highestLayer;
            std::cout << "Updated entry point to node " << id
                    << " at layer " << highestLayer << std::endl;
        }

        // Safety: in principle vectorStored must be true if we reached here.
        if (!vectorStored)
        {
            vectorStorage->storeVectorToDisk(id, point, sectionKey);
        }

    }

    // Links neighbors for upper layers stored in memory
    void HNSWGraph::linkNeighbors(node_id_t id, const std::vector<node_id_t> &neighbors, int layer)
    {
        for (node_id_t neighbor : neighbors)
        {
            nodes[neighbor].neighbors[layer].push_back(id);
            nodes[id].neighbors[layer].push_back(neighbor);
        }
    }

    void HNSWGraph::linkNeighborsAsterDB(node_id_t id, const std::vector<node_id_t> &neighbors)
    {
        db_->AddVertex(id);

        for (node_id_t neighbor : neighbors)
        {
            db_->AddEdge(id, neighbor);
            db_->AddEdge(neighbor, id);
        }
    }

    std::vector<node_id_t> HNSWGraph::selectNeighbors(
        const std::vector<float> &point,
        const std::vector<node_id_t> &candidates,
        int M,
        int layer)
    {
        if (!useHeuristicNeighborSelection_) {
            return selectNeighborsSimple(point, candidates, M, layer);
        } else {
            return selectNeighborsHeuristic2(point, candidates, M, layer);
        }
    }

    // Selects neighbors based on distance and pruning logic
    std::vector<node_id_t> HNSWGraph::selectNeighborsSimple(const std::vector<float> &point, const std::vector<node_id_t> &candidates, int M, int layer)
    {

        if (candidates.size() <= M)
        {
            return candidates;
        }
        else
        {
            std::priority_queue<std::pair<float, node_id_t>> topCandidates;
            for (node_id_t candidate : candidates)
            {
                float dist = 0.0;
                if (layer > 0)
                {
                    dist = euclideanDistance(point, nodes[candidate].point);
                }
                else if (layer == 0)
                {
                    std::vector<float> candidateVector;
                    vectorStorage->readVectorFromDisk(candidate, candidateVector);
                    dist = euclideanDistance(point, candidateVector);
                }

                topCandidates.emplace(dist, candidate);
                if (topCandidates.size() > M)
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
        const std::vector<float> &point,
        const std::vector<node_id_t> &candidates,
        int M,
        int layer)
    {
        // If there are not enough candidates, behave like original code.
        if (candidates.size() <= static_cast<size_t>(M)) {
            return candidates;
        }

        struct CandidateInfo {
            node_id_t   id;
            float distToQuery;
        };

        std::vector<CandidateInfo> candInfos;
        candInfos.reserve(candidates.size());

        // Optional cache for layer 0 to avoid reading the same vectors multiple times.
        // For layer > 0 we can use nodes[id].point directly.
        std::unordered_map<node_id_t, std::vector<float>> vecCache;

        auto getVector = [&](node_id_t id) -> const std::vector<float>& {
            if (layer > 0) {
                return nodes[id].point;
            } else {
                auto it = vecCache.find(id);
                if (it != vecCache.end()) return it->second;

                std::vector<float> v;
                vectorStorage->readVectorFromDisk(id, v);

                auto res = vecCache.emplace(id, std::move(v));
                return res.first->second;
            }
        };

        // 1) Precompute distances query -> candidate
        for (node_id_t id : candidates)
        {
            const auto& candVec = getVector(id);
            float d = euclideanDistance(point, candVec);
            candInfos.push_back(CandidateInfo{id, d});
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
        selected.reserve(M);
        std::vector<node_id_t> rejected;

        for (const auto& cand : candInfos)
        {
            if (selected.size() >= static_cast<size_t>(M))
                break;

            node_id_t id = cand.id;
            float dist_q = cand.distToQuery;

            bool good = true;
            for (node_id_t sid : selected)
            {
                const auto& v1 = getVector(sid);
                const auto& v2 = getVector(id);
                float curdist = euclideanDistance(v1, v2);

                if (curdist < dist_q)
                {
                    good = false;
                    break;
                }
            }

            if (good)
                selected.push_back(id);
            else
                rejected.push_back(id);
        }

        // 4) If we still have fewer than M neighbors, fill from rejected,
        //    preserving the order by distToQuery (because 'rejected' was
        //    filled while scanning candInfos in sorted order).
        for (node_id_t id : rejected)
        {
            if (selected.size() >= static_cast<size_t>(M))
                break;
            selected.push_back(id);
        }

        return selected;
    }

    std::vector<node_id_t> HNSWGraph::selectNeighborsHeuristic2(
        const std::vector<float> &point,
        const std::vector<node_id_t> &candidates,
        int M,
        int layer)
    {
        // If there are not enough candidates, behave like original code.
        if (candidates.size() <= static_cast<size_t>(M)) {
            return candidates;
        }

        struct CandidateInfo {
            node_id_t   id;
            float distToQuery;
        };

        std::vector<CandidateInfo> candInfos;
        candInfos.reserve(candidates.size());

        // Optional cache for layer 0 to avoid reading the same vectors multiple times.
        // For layer > 0 we can use nodes[id].point directly.
        std::unordered_map<node_id_t, std::vector<float>> vecCache;

        auto getVector = [&](node_id_t id) -> const std::vector<float>& {
            if (layer > 0) {
                return nodes[id].point;
            } else {
                auto it = vecCache.find(id);
                if (it != vecCache.end()) return it->second;

                std::vector<float> v;
                vectorStorage->readVectorFromDisk(id, v);

                auto res = vecCache.emplace(id, std::move(v));
                return res.first->second;
            }
        };

        // 1) Precompute distances query -> candidate
        for (node_id_t id : candidates)
        {
            const auto& candVec = getVector(id);
            float d = euclideanDistance(point, candVec);
            candInfos.push_back(CandidateInfo{id, d});
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
        selected.reserve(M);
        std::vector<node_id_t> rejected;

        for (const auto& cand : candInfos)
        {
            if (selected.size() >= static_cast<size_t>(M))
                break;

            node_id_t id = cand.id;
            float dist_q = cand.distToQuery;

            bool good = true;
            for (node_id_t sid : selected)
            {
                const auto& v1 = getVector(sid);
                const auto& v2 = getVector(id);
                float curdist = euclideanDistance(v1, v2);

                if (curdist < dist_q)
                {
                    good = false;
                    break;
                }
            }

            if (good)
                selected.push_back(id);
            else
                rejected.push_back(id);
        }

        return selected;
    }

    std::vector<node_id_t> HNSWGraph::searchLayer(const std::vector<float>& queryPoint,
                                             node_id_t entryPoint,
                                             int ef,
                                             int layer)
    {
        if (layer < 0) {
            std::cerr << "Error: Invalid layer for search.\n";
            return {};
        }
        if (ef <= 0) {
            return {};
        }

        // Visited set
        std::unordered_set<node_id_t> visited;
        visited.reserve(static_cast<std::size_t>(ef) * 4);

        // Candidates: max-heap by (-distance) => smallest distance comes first
        using Cand = std::pair<float, node_id_t>;
        std::priority_queue<Cand> candidates;

        // W: max-heap by (distance) => farthest among current best is on top
        std::priority_queue<Cand> nearest;

        auto get_distance = [&](node_id_t id) -> float {
            if (layer > 0) {
                // Upper layers: vectors are in-memory
                return euclideanDistance(queryPoint, nodes.at(id).point);
            } else {
                // Level 0: vectors are on disk
                std::vector<float> v;
                vectorStorage->readVectorFromDisk(id, v);
                return euclideanDistance(queryPoint, v);
            }
        };

        // Initialize with entry point
        float dist_ep = get_distance(entryPoint);
        visited.insert(entryPoint);
        candidates.emplace(-dist_ep, entryPoint);
        nearest.emplace(dist_ep, entryPoint);

        while (!candidates.empty()) {
            node_id_t current = candidates.top().second;
            float cur_dist = -candidates.top().first;
            candidates.pop();

            // Early termination: if closest candidate is worse than the worst in W, stop
            if (!nearest.empty() && cur_dist > nearest.top().first) {
                break;
            }

            if (layer > 0) {
                // Upper layers: adjacency is in memory.
                const auto node_it = nodes.find(current);
                if (node_it == nodes.end()) {
                    continue; // Defensive: should not happen
                }

                const auto& nbr_map = node_it->second.neighbors;
                auto it = nbr_map.find(layer);
                if (it == nbr_map.end()) {
                    continue; // No adjacency list at this layer (do not create it)
                }

                const auto& nbrs = it->second;
                for (node_id_t nb : nbrs) {
                    if (visited.insert(nb).second) {
                        float d = euclideanDistance(queryPoint, nodes.at(nb).point);
                        if (static_cast<int>(nearest.size()) < ef || d < nearest.top().first) {
                            candidates.emplace(-d, nb);
                            nearest.emplace(d, nb);
                            if (static_cast<int>(nearest.size()) > ef) {
                                nearest.pop();
                            }
                        }
                    }
                }
            } else {
                // Level 0: adjacency is stored in RocksGraph.
                rocksdb::Edges edges;
                db_->GetAllEdges(current, &edges);

                for (uint32_t i = 0; i < edges.num_edges_out; ++i) {
                    node_id_t nb = static_cast<node_id_t>(edges.nxts_out[i].nxt);

                    if (visited.insert(nb).second) {
                        std::vector<float> nb_vec;
                        vectorStorage->readVectorFromDisk(nb, nb_vec);

                        float d = euclideanDistance(queryPoint, nb_vec);
                        if (static_cast<int>(nearest.size()) < ef || d < nearest.top().first) {
                            candidates.emplace(-d, nb);
                            nearest.emplace(d, nb);
                            if (static_cast<int>(nearest.size()) > ef) {
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
    node_id_t HNSWGraph::KNNsearch(const std::vector<float> &queryPoint)
    {
        // W ← ∅ set for the current nearest elements
        std::vector<node_id_t> nearestNeighbors; // W: dynamic list of found nearest neighbors

        // ep ← get enter point for hnsw
        node_id_t currentEntryPoint = entryPoint;
        int currentLayer = maxLayer;

        for (int l = maxLayer; l >= 1; --l)
        {
            std::vector<node_id_t> nearestNeighbors = searchLayer(queryPoint, currentEntryPoint, 30, l);
            currentEntryPoint = nearestNeighbors[0];
        }
        nearestNeighbors = searchLayer(queryPoint, currentEntryPoint, 30, 0);

        return nearestNeighbors[0];
    }

    void HNSWGraph::printState() const
    {
        // We do not print layer 0 by request.
        if (maxLayer <= 0) {
            std::cout << "HNSW state: max_layer=" << maxLayer
                    << " (no upper layers to report)\n";
            return;
        }

        // Count how many nodes have adjacency info at each upper layer.
        // Note: this counts "nodes that currently have neighbor entries at layer l",
        // which is derivable from existing in-memory structures without extra metadata.
        std::vector<std::size_t> layer_node_counts(static_cast<std::size_t>(maxLayer + 1), 0);

        // To avoid double counting, we track per-layer seen node IDs.
        std::vector<std::unordered_set<node_id_t>> seen(static_cast<std::size_t>(maxLayer + 1));

        for (const auto& kv : nodes) {
            node_id_t nid = kv.first;
            const Node& node = kv.second;

            for (const auto& kv2 : node.neighbors) {
                // The key type should be "layer index".
                // If your neighbors map key is int, this is already int.
                // If it's not int, cast safely.
                int layer = static_cast<int>(kv2.first);

                if (layer <= 0) continue;          // skip layer 0 (and any invalid)
                if (layer > maxLayer) continue;    // defensive

                // Option A (default): count node if it has the layer key at all.
                // Option B: count only if the adjacency list is non-empty:
                // if (kv2.second.empty()) continue;

                if (seen[static_cast<std::size_t>(layer)].insert(nid).second) {
                    layer_node_counts[static_cast<std::size_t>(layer)]++;
                }
            }
        }

        std::cout << "HNSW state:\n";
        std::cout << "  max_layer: " << maxLayer << "\n";
        std::cout << "  upper_layers: [1.." << maxLayer << "]\n";

        // Print top-down for readability
        for (int l = maxLayer; l >= 1; --l) {
            std::cout << "  layer " << l << ": "
                    << layer_node_counts[static_cast<std::size_t>(l)]
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