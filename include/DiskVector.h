#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstring>

class VectorStorage
{
private:
    std::string filePath;
    size_t vectorSize;   
    size_t totalVectors; 
    size_t fileSize;     

    std::fstream fileStream; 

public:
    VectorStorage(const std::string &path, size_t vecSize, size_t numVectors)
        : filePath(path), vectorSize(vecSize), totalVectors(numVectors)
    {
        fileSize = vectorSize * totalVectors * sizeof(float);
        fileStream.open(filePath, std::ios::in | std::ios::out | std::ios::binary);
        if (!fileStream.is_open())
        {
            fileStream.open(filePath, std::ios::out | std::ios::binary);
            if (!fileStream.is_open())
            {
                throw std::runtime_error("Failed to create file.");
            }
            fileStream.close();

            fileStream.open(filePath, std::ios::in | std::ios::out | std::ios::binary);
            if (!fileStream.is_open())
            {
                throw std::runtime_error("Failed to open file.");
            }
        }

        fileStream.seekp(0, std::ios::end);
        size_t currentSize = fileStream.tellp();
        if (currentSize < fileSize)
        {
            fileStream.seekp(fileSize - 1);
            fileStream.write("", 1); 
        }
    }

    ~VectorStorage()
    {
        if (fileStream.is_open())
        {
            fileStream.close();
        }
    }

    void storeVectorToDisk(int id, const std::vector<float> &vector)
    {
        if (id < 0 || id >= totalVectors)
        {
            throw std::out_of_range("Vector ID out of range.");
        }
        if (vector.size() != vectorSize)
        {
            throw std::runtime_error("Vector size mismatch.");
        }

        size_t offset = id * vectorSize * sizeof(float);
        fileStream.seekp(offset, std::ios::beg);

        fileStream.write(reinterpret_cast<const char *>(vector.data()), vectorSize * sizeof(float));
        if (!fileStream.good())
        {
            throw std::runtime_error("Failed to write vector to file.");
        }
    }

    void readVectorFromDisk(int id, std::vector<float> &vector)
    {
        if (id < 0 || id >= totalVectors)
        {
            throw std::out_of_range("Vector ID out of range.");
        }

        if (vector.size() != vectorSize)
        {
            vector.resize(vectorSize);
        }

        size_t offset = id * vectorSize * sizeof(float);
        fileStream.seekg(offset, std::ios::beg);

        fileStream.read(reinterpret_cast<char *>(vector.data()), vectorSize * sizeof(float));
        if (!fileStream.good())
        {
            throw std::runtime_error("Failed to read vector from file.");
        }
    }
};