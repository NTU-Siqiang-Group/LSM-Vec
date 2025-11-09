#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "HNSWUtils.h"
#include "HNSWGraph.h"
#include "rocksdb/statistics.h"

int main(int argc, char* argv[])
{
    ROCKSDB_NAMESPACE::Options options;
    options.create_if_missing = true;

    std::string custom_db_path = "/home/shurui/demo";
    uint64_t target_size = 107374182400;
    options.db_paths.emplace_back(ROCKSDB_NAMESPACE::DbPath(custom_db_path, target_size));
    options.statistics = rocksdb::CreateDBStatistics();

    ROCKSDB_NAMESPACE::RocksGraph asterDB(options);
    asterDB.edge_update_policy_ = EDGE_UPDATE_EAGER;

    std::ofstream outFile("output.txt");
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <M> <Mmax> <Ml> <efConstruction>" << std::endl;
        return 1;
    }

    int M = std::stoi(argv[1]);
    int Mmax = std::stoi(argv[2]);
    int Ml = std::stoi(argv[3]);
    float efConstruction = std::stof(argv[4]);
    std::string vectorfilePath = "/home/shurui/demo/vec";

    // siftsmall
    std::string inputfilePath = "/home/shurui/LSMVEC/dataset/siftsmall/siftsmall_base.fvecs";
    std::string queryfilePath = "/home/shurui/LSMVEC/dataset/siftsmall/siftsmall_query.fvecs";
    std::string groundTruthfilePath = "/home/shurui/LSMVEC/dataset/siftsmall/siftsmall_groundtruth.ivecs";

    
    int dim = getdim(inputfilePath);
    ROCKSDB_NAMESPACE::HNSWGraph hnsw(M, Mmax, Ml, efConstruction, &asterDB, outFile, vectorfilePath, dim);

    std::cout << "Inserting nodes from" << inputfilePath << std::endl;
    insertFromFile(hnsw, inputfilePath);
    //hnsw.printStatistics();
    std::cout << "Querying and comparing with ground truth "<< queryfilePath << std::endl;
    queryAndCompareWithGroundTruth(hnsw, queryfilePath, groundTruthfilePath);
    

    return 0;
}
