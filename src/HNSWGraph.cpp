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

    HNSWGraph::HNSWGraph(int M, int Mmax, int Ml, float efConstruction, RocksGraph *db, std::ostream &outFile, std::string vectorfilePath, int vectordim)
        : M(M), Mmax(Mmax), Ml(Ml), efConstruction(efConstruction), db(db), outFile(outFile), gen(rd()), dist(0, 1), maxLayer(-1), entryPoint(-1), ioCount(0), ioTime(0.0), indexingTime(0.0)
    {
        gen.seed(12345);
        vectorStorage = std::make_unique<VectorStorage>(vectorfilePath, vectordim, 100000000);
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
    void HNSWGraph::insertNode(int id, const std::vector<float> &point)
    {
        auto start = std::chrono::high_resolution_clock::now();

        auto vecstart = std::chrono::high_resolution_clock::now();
        vectorStorage->storeVectorToDisk(id, point);
        auto vecend = std::chrono::high_resolution_clock::now();
        double vecduration = std::chrono::duration<double>(vecend - vecstart).count();
        vecwritetime += vecduration; 
        vecwritecount++;             

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
            indexingTime += std::chrono::duration<double>(end - start).count();
            return;
        }

        int currentEntryPoint = entryPoint;

        auto start1 = std::chrono::high_resolution_clock::now(); // Start timing
        for (int l = maxLayer; l > highestLayer; --l)
        {
            std::vector<int> closest = searchLayer(point, currentEntryPoint, 1, l);
            if (!closest.empty())
            {
                currentEntryPoint = closest[0];
            }
        }
        auto end1 = std::chrono::high_resolution_clock::now(); // End timing
        // std::cout << "Search layer took " << std::chrono::duration<double>(end1 - start1).count() << " seconds" << std::endl;

        for (int l = std::min(maxLayer, highestLayer); l >= 0; --l)
        {
            auto start2 = std::chrono::high_resolution_clock::now(); // Start timing
            std::vector<int> neighbors = searchLayer(point, currentEntryPoint, efConstruction, l);
            auto end2 = std::chrono::high_resolution_clock::now(); // End timing
            // std::cout << "search Layer took " << std::chrono::duration<double>(end2 - start2).count() << " seconds" << std::endl;
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
            auto start3 = std::chrono::high_resolution_clock::now(); // Start timing
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
                    auto start = std::chrono::high_resolution_clock::now(); // Start timing
                    db->GetAllEdges(neighbor, &edges);

                    auto end = std::chrono::high_resolution_clock::now(); // End timing
                    double duration = std::chrono::duration<double>(end - start).count();
                    ioTime += duration;     // Accumulate total IO time
                    readIOTime += duration; // Accumulate read IO time
                    ioCount++;
                    readIOCount++; // Increase read IO count

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

                        auto start = std::chrono::high_resolution_clock::now(); // Start timing
                        vectorStorage->readVectorFromDisk(neighbor, cur_point2);
                        auto end = std::chrono::high_resolution_clock::now(); // End timing
                        double duration = std::chrono::duration<double>(end - start).count();
                        vecreadtime += duration; // Accumulate read vector time
                        vecreadcount++;          // Increase read vector count

                        std::vector<int> eNewConn = selectNeighbors(cur_point2, eConns, Mmax, l);
                        // std::cout << "eConns size: " << eConns.size() << std::endl;
                        // std::cout << "eNewConn size: " << eNewConn.size() << std::endl;
                        // set neighbourhood(e) at layer l to eNewConn
                        for (auto node : eConns)
                        {
                            if (std::find(eNewConn.begin(), eNewConn.end(), node) == eNewConn.end())
                            {
                                auto start = std::chrono::high_resolution_clock::now(); // Start timing
                                db->DeleteEdge(neighbor, node);
                                db->DeleteEdge(node, neighbor);
                                auto end = std::chrono::high_resolution_clock::now(); // End timing
                                double duration = std::chrono::duration<double>(end - start).count();
                                ioTime += duration;           // Accumulate total IO time
                                deleteedgeIOTime += duration; // Accumulate write IO time
                                ioCount += 2;
                                deleteedgeIOCount += 2; // Increase write IO count
                            }
                        }
                        // rocksdb::Edges edges;
                        // db->GetAllEdges(neighbor, &edges);
                        // std::cout << "edges.num_edges_out: " << edges.num_edges_out << std::endl;
                    }
                }
            }
            auto end3 = std::chrono::high_resolution_clock::now(); // End timing
            // std::cout << "Shrink connections took " << std::chrono::duration<double>(end3 - start3).count() << " seconds" << std::endl;
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

        auto end = std::chrono::high_resolution_clock::now();
        // std::cout << "Inserting node ID: " << id << " with point size: " << point.size() << " took " << std::chrono::duration<double>(end - start).count() << " seconds" << std::endl;
        indexingTime += std::chrono::duration<double>(end - start).count();
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
        //db->edge_update_policy_ = EDGE_UPDATE_LAZY;
        auto start = std::chrono::high_resolution_clock::now(); // Start timing
        db->AddVertex(nodeId);
        auto end = std::chrono::high_resolution_clock::now(); // End timing
        double duration = std::chrono::duration<double>(end - start).count();
        ioTime += duration;          // Accumulate total IO time
        writenodeIOTime += duration; // Accumulate write IO time
        ioCount++;
        writenodeIOCount++; // Increase write IO count

        // Store point as a property
        /*
        std::string vectorStr;
        for (float val : point)
        {
            vectorStr += std::to_string(val) + " ";
        }
        Property prop{"vector", vectorStr};
        db->AddVertexProperty(nodeId, prop);
        */
        // std::cout << "neighbors size: " << neighbors.size() << std::endl;
        for (int neighbor : neighbors)
        {
            auto start = std::chrono::high_resolution_clock::now(); // Start timing
            db->AddEdge(nodeId, neighbor);
            db->AddEdge(neighbor, nodeId);
            auto end = std::chrono::high_resolution_clock::now(); // End timing
            double duration = std::chrono::duration<double>(end - start).count();
            ioTime += duration;        // Accumulate total IO time
            addedgeIOTime += duration; // Accumulate write IO time
            addedgeIOCount += 2;       // Increase write IO count
            ioCount += 2;
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
                    auto start = std::chrono::high_resolution_clock::now(); // Start timing
                    vectorStorage->readVectorFromDisk(candidate, candidateVector);
                    auto end = std::chrono::high_resolution_clock::now(); // End timing
                    double duration = std::chrono::duration<double>(end - start).count();
                    vecreadtime += duration; // Accumulate read vector time
                    vecreadcount++;          // Increase read vector count
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
                auto start = std::chrono::high_resolution_clock::now();
                vectorStorage->readVectorFromDisk(id, v);
                auto end = std::chrono::high_resolution_clock::now();
                double duration = std::chrono::duration<double>(end - start).count();
                vecreadtime += duration;
                vecreadcount++;

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
                auto start = std::chrono::high_resolution_clock::now();
                vectorStorage->readVectorFromDisk(id, v);
                auto end = std::chrono::high_resolution_clock::now();
                double duration = std::chrono::duration<double>(end - start).count();
                vecreadtime += duration;
                vecreadcount++;

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

            auto start = std::chrono::high_resolution_clock::now(); // Start timing
            vectorStorage->readVectorFromDisk(entryPoint, entryPointVector);
            auto end = std::chrono::high_resolution_clock::now(); // End timing
            double duration = std::chrono::duration<double>(end - start).count();
            vecreadtime += duration; // Accumulate read vector time
            vecreadcount++;          // Increase read vector count

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
                auto start = std::chrono::high_resolution_clock::now(); // Start timing
                db->GetAllEdges(current, &edges);
                auto end = std::chrono::high_resolution_clock::now(); // End timing
                double duration = std::chrono::duration<double>(end - start).count();
                ioTime += duration;     // Accumulate total IO time
                readIOTime += duration; // Accumulate read IO time
                ioCount++;
                readIOCount++; // Increase read IO count

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
                        auto start = std::chrono::high_resolution_clock::now(); // Start timing
                        vectorStorage->readVectorFromDisk(neighbor, neighborVector);
                        auto end = std::chrono::high_resolution_clock::now(); // End timing
                        double duration = std::chrono::duration<double>(end - start).count();
                        vecreadtime += duration; // Accumulate read vector time
                        vecreadcount++;          // Increase read vector count

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

    void HNSWGraph::printStatistics() const
    {
        std::cout << "Indexing Time: " << indexingTime << " seconds" << std::endl;

        std::cout << "-------graph part------" << std::endl;
        std::cout << "Total Aster I/O Operations: " << ioCount << std::endl;
        std::cout << "Total Aster I/O Time: " << ioTime << " seconds" << std::endl;
        std::cout << "Read Operations: " << readIOCount << ", Time: " << readIOTime << " seconds" << std::endl;
        std::cout << "Write Node Operations: " << writenodeIOCount << ", Time: " << writenodeIOTime << " seconds" << std::endl;
        std::cout << "Add Edge Operations: " << addedgeIOCount << ", Time: " << addedgeIOTime << " seconds" << std::endl;
        std::cout << "Delete Edge Operations: " << deleteedgeIOCount << ", Time: " << deleteedgeIOTime << " seconds" << std::endl;
        std::cout << "-------vector part------" << std::endl;
        std::cout << "Total Vector I/O Time: " << vecreadtime + vecwritetime << " seconds" << std::endl;
        std::cout << "Vector Read Operations: " << vecreadcount << ", Time: " << vecreadtime << " seconds" << std::endl;
        std::cout << "Vector Write Operations: " << vecwritecount << ", Time: " << vecwritetime << " seconds" << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }
} // namespace ROCKSDB_NAMESPACE