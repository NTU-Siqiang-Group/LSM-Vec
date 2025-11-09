#include "HNSWGraph.h"
#include <cmath>
#include <limits>
#include <queue>
#include <algorithm>

namespace ROCKSDB_NAMESPACE
{

HNSWGraph::HNSWGraph(int M, int Mmax, int Ml, float efConstruction, RocksGraph *db,
                     std::ostream &outFile, std::string vectorfilePath, int vectordim,
                     bool enableSampling, bool enableReordering)
    : M(M), Mmax(Mmax), Ml(Ml), efConstruction(efConstruction), db(db), outFile(outFile),
      gen(rd()), dist(0, 1), maxLayer(-1), entryPoint(-1), 
      ioCount(0), ioTime(0.0), indexingTime(0.0),
      enableSampling(enableSampling), enableReordering(enableReordering)
{
    gen.seed(12345);
    size_t cacheSize = 10000;
    vectorStorage = std::make_unique<VectorStorage>(vectorfilePath, vectordim, 100000000, cacheSize);
    this->vectordim = vectordim;
    
    if (enableSampling)
    {
        sampler = std::make_unique<SimHashSampler>(hashBits, vectordim);
    }
    
    if (enableReordering)
    {
        reorderer = std::make_unique<GraphReorderer>(reorderWindowSize, reorderLambda);
    }
}


void HNSWGraph::insertNode(int id, const std::vector<float> &point)
{
    auto start = std::chrono::high_resolution_clock::now();

    auto vecstart = std::chrono::high_resolution_clock::now();
    vectorStorage->storeVectorToDisk(id, point);
    auto vecend = std::chrono::high_resolution_clock::now();
    vecwritetime += std::chrono::duration<double>(vecend - vecstart).count();
    vecwritecount++;

    if (enableSampling && sampler)
    {
        sampler->storeHash(id, point);
    }

    int highestLayer = randomLevel();

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
        
        if (enableReordering && reorderer)
        {
            reorderer->registerNode(id, id);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        indexingTime += std::chrono::duration<double>(end - start).count();
        return;
    }

    int currentEntryPoint = entryPoint;
    int sectionEntryPoint = entryPoint;
    
    for (int l = maxLayer; l > highestLayer; --l)
    {
        std::vector<int> closest = searchLayer(point, currentEntryPoint, 1, l);
        if (!closest.empty())
        {
            currentEntryPoint = closest[0];
            
            if (l == 1)  
            {
                sectionEntryPoint = currentEntryPoint;
                std::cout << "[Section] Node " << id 
                          << " will join section of entry point " 
                          << sectionEntryPoint << std::endl;
            }
        }
    }

    for (int l = std::min(maxLayer, highestLayer); l >= 0; --l)
    {
        std::vector<int> neighbors;
        
        if (l == 0 && enableSampling && sampler)
        {
            auto queryHash = sampler->computeHash(point);
            neighbors = searchLayerWithSampling(point, currentEntryPoint, 
                                               efConstruction, l, &queryHash);
        }
        else
        {
            neighbors = searchLayer(point, currentEntryPoint, efConstruction, l);
        }
        
        std::vector<int> selectedNeighbors = selectNeighbors(point, neighbors, M, l);

        if (l > 0)
        {
            linkNeighbors(id, selectedNeighbors, l);
        }
        else if (l == 0)
        {
            linkNeighborsAsterDB(id, point, selectedNeighbors);
            
            if (enableReordering && reorderer)
            {
                reorderer->registerNode(id, sectionEntryPoint);
                std::cout << "[Section] Registered node " << id 
                          << " to section " << sectionEntryPoint << std::endl;
            }
        }

        if (l > 0)
        {
            for (int neighbor : selectedNeighbors)
            {
                std::vector<int> eConn = nodes[neighbor].neighbors[l];
                if (eConn.size() > Mmax)
                {
                    std::vector<int> eNewConn = selectNeighbors(nodes[neighbor].point, 
                                                                eConn, Mmax, l);
                    nodes[neighbor].neighbors[l] = eNewConn;
                }
            }
        }
        else if (l == 0)
        {
            for (int neighbor : selectedNeighbors)
            {
                rocksdb::Edges edges;
                auto start_io = std::chrono::high_resolution_clock::now();
                db->GetAllEdges(neighbor, &edges);
                auto end_io = std::chrono::high_resolution_clock::now();
                ioTime += std::chrono::duration<double>(end_io - start_io).count();
                readIOTime += std::chrono::duration<double>(end_io - start_io).count();
                ioCount++;
                readIOCount++;

                if (edges.num_edges_out > Mmax)
                {
                    std::vector<int> eConns;
                    for (uint32_t i = 0; i < edges.num_edges_out; ++i)
                    {
                        eConns.push_back(edges.nxts_out[i].nxt);
                    }
                    
                    std::vector<float> cur_point;
                    auto start_vec = std::chrono::high_resolution_clock::now();
                    vectorStorage->readVectorFromDisk(neighbor, cur_point);
                    auto end_vec = std::chrono::high_resolution_clock::now();
                    vecreadtime += std::chrono::duration<double>(end_vec - start_vec).count();
                    vecreadcount++;

                    std::vector<int> eNewConn = selectNeighbors(cur_point, eConns, Mmax, l);

                    for (auto node : eConns)
                    {
                        if (std::find(eNewConn.begin(), eNewConn.end(), node) == eNewConn.end())
                        {
                            auto start_del = std::chrono::high_resolution_clock::now();
                            db->DeleteEdge(neighbor, node);
                            db->DeleteEdge(node, neighbor);
                            auto end_del = std::chrono::high_resolution_clock::now();
                            deleteedgeIOTime += std::chrono::duration<double>(end_del - start_del).count();
                            ioTime += std::chrono::duration<double>(end_del - start_del).count();
                            ioCount += 2;
                            deleteedgeIOCount += 2;
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
        maxLayer = highestLayer;
        std::cout << "Updated entry point to node " << id 
                  << " at layer " << highestLayer << std::endl;
    }

    if (enableReordering)
    {
        insertionsSinceReorder++;
        maybeReorder();
    }

    auto end = std::chrono::high_resolution_clock::now();
    indexingTime += std::chrono::duration<double>(end - start).count();
}


void HNSWGraph::printSectionStatistics() const
{
    if (!enableReordering || !reorderer)
    {
        return;
    }
    
    std::cout << "\n=== Section Statistics ===" << std::endl;
    reorderer->printStatistics();
}
std::vector<int> HNSWGraph::searchLayerWithSampling(
    const std::vector<float> &queryPoint,
    int entryPoint,
    int ef,
    int layer,
    const std::vector<int8_t> *queryHash)
{
    std::unordered_set<int> visited;
    std::priority_queue<std::pair<float, int>> candidates;
    std::priority_queue<std::pair<float, int>> nearestNeighbors;

    std::vector<float> entryPointVector;
    auto start = std::chrono::high_resolution_clock::now();
    vectorStorage->readVectorFromDisk(entryPoint, entryPointVector);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    vecreadtime += duration;
    vecreadcount++;

    float distToEP = euclideanDistance(queryPoint, entryPointVector);
    visited.insert(entryPoint);
    candidates.emplace(-distToEP, entryPoint);
    nearestNeighbors.emplace(distToEP, entryPoint);

    int collisionThreshold = hashBits / 2;
    if (sampler && queryHash)
    {
        float delta = nearestNeighbors.empty() ? 1.0f : nearestNeighbors.top().first;
        collisionThreshold = sampler->computeCollisionThreshold(epsilon, delta);
    }

    while (!candidates.empty())
    {
        int current = candidates.top().second;
        float currentDist = -candidates.top().first;
        candidates.pop();

        if (currentDist > nearestNeighbors.top().first)
        {
            break;
        }

        rocksdb::Edges edges;
        auto start_io = std::chrono::high_resolution_clock::now();
        db->GetAllEdges(current, &edges);
        auto end_io = std::chrono::high_resolution_clock::now();
        double duration_io = std::chrono::duration<double>(end_io - start_io).count();
        ioTime += duration_io;
        readIOTime += duration_io;
        ioCount++;
        readIOCount++;

        std::vector<int> neighbors;
        for (uint32_t i = 0; i < edges.num_edges_out; ++i)
        {
            neighbors.push_back(edges.nxts_out[i].nxt);
        }

        std::vector<int> sampledNeighbors;
        if (sampler && queryHash && samplingRatio < 1.0f)
        {
            sampledNeighbors = sampler->sampleNeighbors(*queryHash, neighbors, 
                                                        samplingRatio, collisionThreshold);
        }
        else
        {
            sampledNeighbors = neighbors;
        }

        if (!sampledNeighbors.empty())
        {
            vectorStorage->prefetchVectors(sampledNeighbors);
        }

        for (int neighbor : sampledNeighbors)
        {
            if (visited.find(neighbor) == visited.end())
            {
                visited.insert(neighbor);

                std::vector<float> neighborVector;
                auto start_vec = std::chrono::high_resolution_clock::now();
                vectorStorage->readVectorFromDisk(neighbor, neighborVector);
                auto end_vec = std::chrono::high_resolution_clock::now();
                double duration_vec = std::chrono::duration<double>(end_vec - start_vec).count();
                vecreadtime += duration_vec;
                vecreadcount++;

                float dist = euclideanDistance(queryPoint, neighborVector);
                
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
    std::sort(temp.begin(), temp.end(), 
              [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
                  return a.first < b.first;
              });

    std::vector<int> result;
    for (const auto &pair : temp)
    {
        result.push_back(pair.second);
    }
    
    return result;
}

int HNSWGraph::KNNsearch(const std::vector<float> &queryPoint)
{
    std::vector<int> nearestNeighbors;
    int currentEntryPoint = entryPoint;
    
    std::vector<int8_t> queryHash;
    if (enableSampling && sampler)
    {
        queryHash = sampler->computeHash(queryPoint);
    }

    for (int l = maxLayer; l >= 1; --l)
    {
        nearestNeighbors = searchLayer(queryPoint, currentEntryPoint, 30, l);
        currentEntryPoint = nearestNeighbors[0];
    }
    
    if (enableSampling && sampler)
    {
        nearestNeighbors = searchLayerWithSampling(queryPoint, currentEntryPoint, 
                                                   30, 0, &queryHash);
    }
    else
    {
        nearestNeighbors = searchLayer(queryPoint, currentEntryPoint, 30, 0);
    }

    return nearestNeighbors[0];
}

void HNSWGraph::maybeReorder()
{
    if (!enableReordering || !reorderer)
    {
        return;
    }
    
    if (insertionsSinceReorder >= reorderInterval)
    {
        std::cout << "\n*** Triggering graph reordering after " 
                  << insertionsSinceReorder << " insertions ***" << std::endl;
        reorderGraph();
        insertionsSinceReorder = 0;
    }
}

void HNSWGraph::reorderGraph()
{
    if (!enableReordering || !reorderer)
    {
        return;
    }
    
    std::cout << "Starting graph reordering..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::unordered_map<int, std::vector<int>> edgeList;
    reorderer->reorderAllSections(edgeList);
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - start).count();
    
    std::cout << "Graph reordering completed in " << duration << "s" << std::endl;
}

void HNSWGraph::printStatistics() const
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "LSM-VEC Performance Statistics" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n--- Indexing ---" << std::endl;
    std::cout << "Indexing Time: " << indexingTime << " seconds" << std::endl;

    std::cout << "\n--- Graph I/O ---" << std::endl;
    std::cout << "Total Aster I/O Operations: " << ioCount << std::endl;
    std::cout << "Total Aster I/O Time: " << ioTime << " seconds" << std::endl;
    std::cout << "  Read: " << readIOCount << " ops, " << readIOTime << "s" << std::endl;
    std::cout << "  Write Node: " << writenodeIOCount << " ops, " << writenodeIOTime << "s" << std::endl;
    std::cout << "  Add Edge: " << addedgeIOCount << " ops, " << addedgeIOTime << "s" << std::endl;
    std::cout << "  Delete Edge: " << deleteedgeIOCount << " ops, " << deleteedgeIOTime << "s" << std::endl;
    
    std::cout << "\n--- Vector Storage ---" << std::endl;
    vectorStorage->printStatistics();
    
    if (enableSampling && sampler)
    {
        std::cout << "\n--- SimHash Sampling ---" << std::endl;
        sampler->printStatistics();
        std::cout << "Memory usage: " << (sampler->getMemoryUsage() / 1024.0 / 1024.0) 
                  << " MB" << std::endl;
    }
    
    if (enableReordering && reorderer)
    {
        std::cout << "\n--- Graph Reordering ---" << std::endl;
        reorderer->printStatistics();
    }
    
    std::cout << "\n========================================\n" << std::endl;
}

} // namespace ROCKSDB_NAMESPACE