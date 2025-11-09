#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <iostream>
#include <memory>

namespace ROCKSDB_NAMESPACE
{
    class GraphReorderer
    {
    private:
        struct Section
        {
            int entryPointId;
            std::vector<int> nodeIds;
            std::unordered_map<int, int> nodeToPosition;
            std::unordered_map<std::pair<int, int>, float, 
                             std::hash<std::pair<int, int>>> edgeScores;
        };

        std::unordered_map<int, Section> sections;
        std::unordered_map<int, int> nodeToSection;
        int windowSize;
        float lambda;
        size_t totalReorderings;
        double totalReorderingTime;

    public:
        GraphReorderer(int windowSize = 64, float lambda = 1.0f)
            : windowSize(windowSize), lambda(lambda),
              totalReorderings(0), totalReorderingTime(0.0)
        {
            std::cout << "GraphReorderer initialized with window size: " 
                      << windowSize << ", lambda: " << lambda << std::endl;
        }

        void registerNode(int nodeId, int entryPointId)
        {
            if (sections.find(entryPointId) == sections.end())
            {
                sections[entryPointId].entryPointId = entryPointId;
            }
            
            Section &section = sections[entryPointId];
            section.nodeIds.push_back(nodeId);
            section.nodeToPosition[nodeId] = section.nodeIds.size() - 1;
            nodeToSection[nodeId] = entryPointId;
        }

        void removeNode(int nodeId)
        {
            auto it = nodeToSection.find(nodeId);
            if (it == nodeToSection.end())
            {
                return;
            }
            
            int entryPointId = it->second;
            Section &section = sections[entryPointId];
            
            auto nodeIt = section.nodeToPosition.find(nodeId);
            if (nodeIt != section.nodeToPosition.end())
            {
                int pos = nodeIt->second;
                
                if (pos < section.nodeIds.size() - 1)
                {
                    int lastNode = section.nodeIds.back();
                    section.nodeIds[pos] = lastNode;
                    section.nodeToPosition[lastNode] = pos;
                }
                
                section.nodeIds.pop_back();
                section.nodeToPosition.erase(nodeId);
            }
            
            nodeToSection.erase(it);
        }

        float computeEdgeScore(int u, int v, 
                              int sharedInNeighbors,
                              bool directConnection,
                              int hammingDist = 0)
        {
            float score = 0.0f;
            score += static_cast<float>(sharedInNeighbors);
            
            if (directConnection)
            {
                score += (1.0f + lambda);
            }
            
            if (hammingDist > 0)
            {
                score *= (1.0f / (1.0f + hammingDist));
            }
            
            return score;
        }

        void reorderSection(int entryPointId,
                           const std::unordered_map<int, std::vector<int>> &edgeList,
                           const std::unordered_map<int, int> &hammingDistances = {})
        {
            auto startTime = std::chrono::high_resolution_clock::now();
            
            auto it = sections.find(entryPointId);
            if (it == sections.end() || it->second.nodeIds.empty())
            {
                return;
            }
            
            Section &section = it->second;
            std::vector<int> &nodes = section.nodeIds;
            
            if (nodes.size() <= 1)
            {
                return;
            }
            
            std::unordered_map<std::pair<int, int>, float, 
                             std::hash<std::pair<int, int>>> scores;
            
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                int u = nodes[i];
                auto uEdges = edgeList.find(u);
                if (uEdges == edgeList.end()) continue;
                
                for (size_t j = i + 1; j < std::min(i + windowSize, nodes.size()); ++j)
                {
                    int v = nodes[j];
                    bool connected = false;
                    int sharedNeighbors = 0;
                    
                    auto vEdges = edgeList.find(v);
                    if (vEdges != edgeList.end())
                    {
                        connected = std::find(uEdges->second.begin(), 
                                            uEdges->second.end(), v) != uEdges->second.end();
                        
                        for (int neighbor : uEdges->second)
                        {
                            if (std::find(vEdges->second.begin(), 
                                        vEdges->second.end(), neighbor) != vEdges->second.end())
                            {
                                sharedNeighbors++;
                            }
                        }
                    }
                    
                    int hammingU = 0;
                    auto hIt = hammingDistances.find(u);
                    if (hIt != hammingDistances.end())
                    {
                        hammingU = hIt->second;
                    }
                    
                    float score = computeEdgeScore(u, v, sharedNeighbors, 
                                                  connected, hammingU);
                    
                    if (score > 0.0f)
                    {
                        scores[{u, v}] = score;
                    }
                }
            }
            
            std::vector<int> newOrder;
            std::unordered_set<int> visited;
            
            int startNode = nodes[0];
            int maxDegree = 0;
            for (int node : nodes)
            {
                auto edgeIt = edgeList.find(node);
                if (edgeIt != edgeList.end() && edgeIt->second.size() > maxDegree)
                {
                    maxDegree = edgeIt->second.size();
                    startNode = node;
                }
            }
            
            newOrder.push_back(startNode);
            visited.insert(startNode);
            
            while (newOrder.size() < nodes.size())
            {
                float bestScore = -1.0f;
                int bestNode = -1;
                
                size_t windowStart = newOrder.size() > windowSize ? 
                                    newOrder.size() - windowSize : 0;
                
                for (size_t i = windowStart; i < newOrder.size(); ++i)
                {
                    int u = newOrder[i];
                    
                    for (int v : nodes)
                    {
                        if (visited.find(v) != visited.end()) continue;
                        
                        auto scoreIt = scores.find({std::min(u, v), std::max(u, v)});
                        if (scoreIt != scores.end() && scoreIt->second > bestScore)
                        {
                            bestScore = scoreIt->second;
                            bestNode = v;
                        }
                    }
                }
                
                if (bestNode == -1)
                {
                    for (int node : nodes)
                    {
                        if (visited.find(node) == visited.end())
                        {
                            bestNode = node;
                            break;
                        }
                    }
                }
                
                newOrder.push_back(bestNode);
                visited.insert(bestNode);
            }
            
            section.nodeIds = newOrder;
            for (size_t i = 0; i < newOrder.size(); ++i)
            {
                section.nodeToPosition[newOrder[i]] = i;
            }
            
            auto endTime = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double>(endTime - startTime).count();
            
            totalReorderings++;
            totalReorderingTime += duration;
            
            std::cout << "Reordered section " << entryPointId 
                      << " with " << nodes.size() << " nodes in " 
                      << duration << "s" << std::endl;
        }

        void reorderAllSections(const std::unordered_map<int, std::vector<int>> &edgeList)
        {
            std::cout << "Reordering " << sections.size() << " sections..." << std::endl;
            
            for (auto &entry : sections)
            {
                reorderSection(entry.first, edgeList);
            }
            
            std::cout << "Completed reordering all sections" << std::endl;
        }

        int getPhysicalPosition(int nodeId) const
        {
            auto sectionIt = nodeToSection.find(nodeId);
            if (sectionIt == nodeToSection.end())
            {
                return -1;
            }
            
            auto sectIt = sections.find(sectionIt->second);
            if (sectIt == sections.end())
            {
                return -1;
            }
            
            auto posIt = sectIt->second.nodeToPosition.find(nodeId);
            if (posIt == sectIt->second.nodeToPosition.end())
            {
                return -1;
            }
            
            return posIt->second;
        }

        std::unordered_map<int, int> getMappingTable() const
        {
            std::unordered_map<int, int> mapping;
            int physicalId = 0;
            
            for (const auto &entry : sections)
            {
                const Section &section = entry.second;
                for (int nodeId : section.nodeIds)
                {
                    mapping[nodeId] = physicalId++;
                }
            }
            
            return mapping;
        }

        void printStatistics() const
        {
            std::cout << "=== Graph Reordering Statistics ===" << std::endl;
            std::cout << "Total sections: " << sections.size() << std::endl;
            std::cout << "Total nodes: " << nodeToSection.size() << std::endl;
            std::cout << "Total reorderings: " << totalReorderings << std::endl;
            std::cout << "Total reordering time: " << totalReorderingTime << "s" << std::endl;
            
            if (!sections.empty())
            {
                std::cout << "\nSection sizes:" << std::endl;
                for (const auto &entry : sections)
                {
                    std::cout << "  Section " << entry.first << ": " 
                              << entry.second.nodeIds.size() << " nodes" << std::endl;
                }
            }
        }

        size_t getSectionCount() const
        {
            return sections.size();
        }
    };

} // namespace ROCKSDB_NAMESPACE