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

namespace ROCKSDB_NAMESPACE
{

    HNSWGraph::HNSWGraph(int M, int Mmax, int Ml, float efConstruction, RocksGraph *db, std::ostream &outFile, std::string vectorfilePath, int vectordim, const Config& cfg)
        : M(M), Mmax(Mmax), Ml(Ml), efConstruction(efConstruction), db(db), outFile(outFile), gen(rd()), dist(0, 1), maxLayer(-1), entryPoint(-1)
    {
        gen.seed(12345);
        //vectorStorage = std::make_unique<VectorStorage>(vectorfilePath, vectordim, 100000000);
        //vectorStorage = std::make_unique<VectorStorage>(vectorfilePath, vectordim);
        if (cfg.vector_storage_type == 1) {
            printf("Using page-based vector storage layout\n");
            vectorStorage = std::make_unique<PagedVectorStorage>(
                cfg.vector_file_path,
                static_cast<size_t>(vectordim),
                cfg.vec_file_capacity,
                cfg.paged_max_cached_pages
            );
        } else {
            printf("Using plain vector storage layout\n");
            vectorStorage = std::make_unique<BasicVectorStorage>(
                cfg.vector_file_path,
                static_cast<size_t>(vectordim),
                cfg.vec_file_capacity
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

    // Inserts a node into the HNSW graph
    void HNSWGraph::insertNodeOld(int id, const std::vector<float> &point)
    {
        vectorStorage->storeVectorToDisk(id, point);         

        int highestLayer = randomLevel();
        // std::cout << "Node " << id << " assigned to highest layer: " << highestLayer << std::endl;

        if (highestLayer > 0)
        {
            Node newNode{id, point};
            newNode.neighbors = std::unordered_map<int, std::vector<int>>();
            nodes[id] = newNode;
        }

        if (entryPoint == -1)
        {
            entryPoint = id;
            maxLayer = highestLayer;
            linkNeighborsAsterDB(id, point, {});
            // std::cout << "First node inserted. Entry point set to: " << id << std::endl;

            auto end = std::chrono::high_resolution_clock::now();
            // std::cout << "Inserting node ID: " << id << " with point size: " << point.size() << " took " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;
            return;
        }

        int currentEntryPoint = entryPoint;

        for (int l = maxLayer; l > highestLayer; --l)
        {
            std::vector<int> closest = searchLayer(point, currentEntryPoint, 1, l);
            if (!closest.empty())
            {
                currentEntryPoint = closest[0];
            }
        }

        for (int l = std::min(maxLayer, highestLayer); l >= 0; --l)
        {
            std::vector<int> neighbors = searchLayer(point, currentEntryPoint, efConstruction, l);
            std::vector<int> selectedNeighbors;
            if(l > 0)
                selectedNeighbors = selectNeighbors(point, neighbors, M, l);
            else
                selectedNeighbors = selectNeighbors(point, neighbors, Mmax, l);

            // add bidirectionall connectionts
            if (l > 0)
            {
                linkNeighbors(id, selectedNeighbors, l);
            }
            else if (l == 0)
            {
                linkNeighborsAsterDB(id, point, selectedNeighbors);
            }

            // shrink connections if needed
            if (l > 0)
            {
                for (int neighbor : selectedNeighbors)
                {
                    // eConn ← neighbourhood(e) at layer l
                    std::vector<int> eConn = nodes[neighbor].neighbors[l];
                    if (eConn.size() > M)
                    {
                        std::vector<int> eNewConn = selectNeighbors(nodes[neighbor].point, eConn, M, l);
                        // set neighbourhood(e) at layer l to eNewConn
                        nodes[neighbor].neighbors[l] = eNewConn;
                    }
                }
            }
            else if (l == 0)
            {
                for (int neighbor : selectedNeighbors)
                {
                    // eConn ← neighbourhood(e) at layer l
                    rocksdb::Edges edges;
                    db->GetAllEdges(neighbor, &edges);

                    if (edges.num_edges_out > Mmax)
                    {
                        //db->edge_update_policy_ = EDGE_UPDATE_EAGER;
                        std::vector<int> eConns;
                        for (uint32_t i = 0; i < edges.num_edges_out; ++i)
                        {
                            eConns.push_back(edges.nxts_out[i].nxt);
                        }
                        /*
                        std::vector<rocksdb::Property> props;
                        db->GetVertexProperty(neighbor, props);
                        */
                        //std::vector<float> cur_point1 = nodes[neighbor].point;
                        std::vector<float> cur_point2;

                        vectorStorage->readVectorFromDisk(neighbor, cur_point2);

                        std::vector<int> eNewConn = selectNeighbors(cur_point2, eConns, Mmax, l);
                        // std::cout << "eConns size: " << eConns.size() << std::endl;
                        // std::cout << "eNewConn size: " << eNewConn.size() << std::endl;
                        // set neighbourhood(e) at layer l to eNewConn
                        for (auto node : eConns)
                        {
                            if (std::find(eNewConn.begin(), eNewConn.end(), node) == eNewConn.end())
                            {
                                db->DeleteEdge(neighbor, node);
                                db->DeleteEdge(node, neighbor);
                            }
                        }
                        // rocksdb::Edges edges;
                        // db->GetAllEdges(neighbor, &edges);
                        // std::cout << "edges.num_edges_out: " << edges.num_edges_out << std::endl;
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
            maxLayer = highestLayer;
            std::cout << "Updated entry point to node " << id << " at layer " << highestLayer << std::endl;
        }

        // std::cout << "Inserting node ID: " << id << " with point size: " << point.size() << " took " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;
    }

    void HNSWGraph::insertNode(int id, const std::vector<float> &point)
    {
        bool vectorStored = false;  // NEW: track whether we've stored this vector

        int highestLayer = randomLevel();

        if (highestLayer > 0)
        {
            Node newNode{id, point};
            newNode.neighbors = std::unordered_map<int, std::vector<int>>();
            nodes[id] = newNode;
        }

        // ----- First node special case -----
        if (entryPoint == -1)
        {
            entryPoint = id;
            maxLayer   = highestLayer;

            linkNeighborsAsterDB(id, point, {});

            // For the first node, we can just use id as sectionKey
            int sectionKey = id;
            vectorStorage->storeVectorToDisk(id, point, sectionKey);
            vectorStored = true;

            return;
        }

        int currentEntryPoint = entryPoint;

        // ----- Search down from top layer to (highestLayer+1) to choose entry -----
        for (int l = maxLayer; l > highestLayer; --l)
        {
            std::vector<int> closest = searchLayer(point, currentEntryPoint, 1, l);
            if (!closest.empty())
            {
                currentEntryPoint = closest[0];
            }
        }

        // Initial guess for sectionKey: if we have upper layers, use currentEntryPoint,
        // otherwise fall back to global entryPoint.
        int sectionKey = (maxLayer >= 1 ? currentEntryPoint : entryPoint);

        // ----- From min(maxLayer, highestLayer) ... down to 0 -----
        for (int l = std::min(maxLayer, highestLayer); l >= 0; --l)
        {
            std::vector<int> neighbors =
                searchLayer(point, currentEntryPoint, efConstruction, l);
            std::vector<int> selectedNeighbors =
                selectNeighbors(point, neighbors, M, l);

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
                linkNeighborsAsterDB(id, point, selectedNeighbors);
            }

            // ---- Shrink connections ----
            if (l > 0)
            {
                for (int neighbor : selectedNeighbors)
                {
                    std::vector<int> eConn = nodes[neighbor].neighbors[l];
                    if (eConn.size() > static_cast<size_t>(Mmax))
                    {
                        std::vector<int> eNewConn =
                            selectNeighbors(nodes[neighbor].point, eConn, Mmax, l);
                        nodes[neighbor].neighbors[l] = std::move(eNewConn);
                    }
                }
            }
            else // l == 0
            {
                for (int neighbor : selectedNeighbors)
                {
                    rocksdb::Edges edges;
                    db->GetAllEdges(neighbor, &edges);

                    if (edges.num_edges_out > static_cast<uint32_t>(Mmax))
                    {
                        std::vector<int> eConns;
                        eConns.reserve(edges.num_edges_out);
                        for (uint32_t i = 0; i < edges.num_edges_out; ++i)
                        {
                            eConns.push_back(edges.nxts_out[i].nxt);
                        }

                        std::vector<float> cur_point2;
                        vectorStorage->readVectorFromDisk(neighbor, cur_point2);

                        std::vector<int> eNewConn =
                            selectNeighbors(cur_point2, eConns, Mmax, l);

                        for (auto node : eConns)
                        {
                            if (std::find(eNewConn.begin(), eNewConn.end(), node) == eNewConn.end())
                            {
                                db->DeleteEdge(neighbor, node);
                                db->DeleteEdge(node, neighbor);
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
    void HNSWGraph::linkNeighbors(int nodeid, const std::vector<int> &neighbors, int layer)
    {
        for (int neighbor : neighbors)
        {
            nodes[neighbor].neighbors[layer].push_back(nodeid);
            nodes[nodeid].neighbors[layer].push_back(neighbor);
        }
    }

    void HNSWGraph::linkNeighborsAsterDB(int nodeId, const std::vector<float> &point, const std::vector<int> &neighbors)
    {
        db->AddVertex(nodeId);

        for (int neighbor : neighbors)
        {
            db->AddEdge(nodeId, neighbor);
            db->AddEdge(neighbor, nodeId);
        }
    }

    std::vector<int> HNSWGraph::selectNeighbors(
        const std::vector<float> &point,
        const std::vector<int> &candidates,
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
    std::vector<int> HNSWGraph::selectNeighborsSimple(const std::vector<float> &point, const std::vector<int> &candidates, int M, int layer)
    {

        if (candidates.size() <= M)
        {
            return candidates;
        }
        else
        {
            std::priority_queue<std::pair<float, int>> topCandidates;
            for (int candidate : candidates)
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

            std::vector<int> selectedNeighbors;
            while (!topCandidates.empty())
            {
                selectedNeighbors.push_back(topCandidates.top().second);
                topCandidates.pop();
            }
            return selectedNeighbors;
        }
    }

    std::vector<int> HNSWGraph::selectNeighborsHeuristic1(
        const std::vector<float> &point,
        const std::vector<int> &candidates,
        int M,
        int layer)
    {
        // If there are not enough candidates, behave like original code.
        if (candidates.size() <= static_cast<size_t>(M)) {
            return candidates;
        }

        struct CandidateInfo {
            int   id;
            float distToQuery;
        };

        std::vector<CandidateInfo> candInfos;
        candInfos.reserve(candidates.size());

        // Optional cache for layer 0 to avoid reading the same vectors multiple times.
        // For layer > 0 we can use nodes[id].point directly.
        std::unordered_map<int, std::vector<float>> vecCache;

        auto getVector = [&](int id) -> const std::vector<float>& {
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
        for (int id : candidates)
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
        std::vector<int> selected;
        selected.reserve(M);
        std::vector<int> rejected;

        for (const auto& cand : candInfos)
        {
            if (selected.size() >= static_cast<size_t>(M))
                break;

            int id = cand.id;
            float dist_q = cand.distToQuery;

            bool good = true;
            for (int sid : selected)
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
        for (int id : rejected)
        {
            if (selected.size() >= static_cast<size_t>(M))
                break;
            selected.push_back(id);
        }

        return selected;
    }

    std::vector<int> HNSWGraph::selectNeighborsHeuristic2(
        const std::vector<float> &point,
        const std::vector<int> &candidates,
        int M,
        int layer)
    {
        // If there are not enough candidates, behave like original code.
        if (candidates.size() <= static_cast<size_t>(M)) {
            return candidates;
        }

        struct CandidateInfo {
            int   id;
            float distToQuery;
        };

        std::vector<CandidateInfo> candInfos;
        candInfos.reserve(candidates.size());

        // Optional cache for layer 0 to avoid reading the same vectors multiple times.
        // For layer > 0 we can use nodes[id].point directly.
        std::unordered_map<int, std::vector<float>> vecCache;

        auto getVector = [&](int id) -> const std::vector<float>& {
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
        for (int id : candidates)
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
        std::vector<int> selected;
        selected.reserve(M);
        std::vector<int> rejected;

        for (const auto& cand : candInfos)
        {
            if (selected.size() >= static_cast<size_t>(M))
                break;

            int id = cand.id;
            float dist_q = cand.distToQuery;

            bool good = true;
            for (int sid : selected)
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

    std::vector<int> HNSWGraph::searchLayer(const std::vector<float> &queryPoint, int entryPoint, int ef, int layer)
    {
        // set of visited elements
        std::unordered_set<int> visited;

        // set of candidates
        std::priority_queue<std::pair<float, int>> candidates; // C: set of candidates

        // dynamic list of found nearest neighbors
        std::priority_queue<std::pair<float, int>> nearestNeighbors; // W: dynamic list of found nearest neighbors

        if (layer > 0)
        {
            // initialize the search
            float distToEP = euclideanDistance(queryPoint, nodes[entryPoint].point);
            visited.insert(entryPoint); // v ← ep
            candidates.emplace(-distToEP, entryPoint);
            nearestNeighbors.emplace(distToEP, entryPoint);

            while (!candidates.empty())
            {
                // get the nearest candidate, extract nearest element from C
                int current = candidates.top().second;
                float currentDist = -candidates.top().first;
                candidates.pop();

                // check if the current candidate is closer than the farthest neighbor
                // get furthest element from W to q
                if (currentDist > nearestNeighbors.top().first)
                {
                    break;
                }
                for (int neighbor : nodes[current].neighbors[layer])
                {
                    if (visited.find(neighbor) == visited.end())
                    {
                        visited.insert(neighbor);
                        float dist = euclideanDistance(queryPoint, nodes[neighbor].point);
                        if (nearestNeighbors.size() < ef || dist < nearestNeighbors.top().first)
                        {
                            candidates.emplace(-dist, neighbor);
                            nearestNeighbors.emplace(dist, neighbor);
                            if (nearestNeighbors.size() > ef)
                            {
                                nearestNeighbors.pop();
                            }
                        }
                    }
                }
            }
            std::vector<std::pair<float, int>> temp;

            
            while (!nearestNeighbors.empty())
            {
                temp.push_back(nearestNeighbors.top());
                nearestNeighbors.pop();
            }

            std::sort(temp.begin(), temp.end(), [](const std::pair<float, int> &a, const std::pair<float, int> &b)
                      {
                          return a.first < b.first;
                      });

            std::vector<int> result;
            for (const auto &pair : temp)
            {
                result.push_back(pair.second);
            }
            return result;
        }
        else if (layer == 0)
        {
            // initialize the search
            // float distToEP = euclideanDistance(queryPoint, nodes[entryPoint].point);
            std::vector<float> entryPointVector;

            vectorStorage->readVectorFromDisk(entryPoint, entryPointVector);

            float distToEP = euclideanDistance(queryPoint, entryPointVector);
            visited.insert(entryPoint); // v ← ep
            candidates.emplace(-distToEP, entryPoint);
            nearestNeighbors.emplace(distToEP, entryPoint);

            while (!candidates.empty())
            {
                // extract nearest element from C to q
                // get furthest element from W to q
                int current = candidates.top().second;
                float currentDist = -candidates.top().first;
                candidates.pop();

                // check if the current candidate is closer than the farthest neighbor
                // get furthest element from W to q
                if (currentDist > nearestNeighbors.top().first)
                {
                    break;
                }

                rocksdb::Edges edges;
                db->GetAllEdges(current, &edges);

                std::vector<int> neighbors;
                for (uint32_t i = 0; i < edges.num_edges_out; ++i)
                {
                    neighbors.push_back(edges.nxts_out[i].nxt);
                }

                for (int neighbor : neighbors)
                {
                    if (visited.find(neighbor) == visited.end())
                    {
                        visited.insert(neighbor);
                        // float dist = euclideanDistance(queryPoint, nodes[neighbor].point);
                        std::vector<float> neighborVector;
                        vectorStorage->readVectorFromDisk(neighbor, neighborVector);

                        float dist = euclideanDistance(queryPoint, neighborVector);
                        if (nearestNeighbors.size() < ef || dist < nearestNeighbors.top().first)
                        {
                            candidates.emplace(-dist, neighbor);
                            nearestNeighbors.emplace(dist, neighbor);
                            if (nearestNeighbors.size() > ef)
                            {
                                nearestNeighbors.pop(); // remove furthest element from W to q
                            }
                        }
                    }
                }
            }
            std::vector<std::pair<float, int>> temp;

            while (!nearestNeighbors.empty())
            {
                temp.push_back(nearestNeighbors.top());
                nearestNeighbors.pop();
            }

            std::sort(temp.begin(), temp.end(), [](const std::pair<float, int> &a, const std::pair<float, int> &b)
                      {
                          return a.first < b.first; // 比较距离
                      });

            std::vector<int> result;
            for (const auto &pair : temp)
            {
                result.push_back(pair.second);
            }
            return result;
        }
        else
        {
            // error
            std::cerr << "Error: Invalid layer for search." << std::endl;
        }
    }

    // Performs a greedy search to find the closest neighbor at a specific layer
    int HNSWGraph::KNNsearch(const std::vector<float> &queryPoint)
    {
        // W ← ∅ set for the current nearest elements
        std::vector<int> nearestNeighbors; // W: dynamic list of found nearest neighbors

        // ep ← get enter point for hnsw
        int currentEntryPoint = entryPoint;
        int currentLayer = maxLayer;

        for (int l = maxLayer; l >= 1; --l)
        {
            std::vector<int> nearestNeighbors = searchLayer(queryPoint, currentEntryPoint, 30, l);
            currentEntryPoint = nearestNeighbors[0];
        }
        nearestNeighbors = searchLayer(queryPoint, currentEntryPoint, 30, 0);

        return nearestNeighbors[0];
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