#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "HNSWUtils.h"
#include "HNSWGraph.h"
#include "Config.h"
#include "rocksdb/statistics.h"

int main(int argc, char* argv[])
{
    Config cfg = Config::Parse(argc, argv);

    ROCKSDB_NAMESPACE::Options options;
    options.create_if_missing = true;
    options.db_paths.emplace_back(ROCKSDB_NAMESPACE::DbPath(cfg.db_path, cfg.db_target_size));
    options.statistics = rocksdb::CreateDBStatistics();

    ROCKSDB_NAMESPACE::RocksGraph asterDB(options);
    ApplyEdgeUpdatePolicy(asterDB, cfg.edge_update_policy);

    std::ofstream outFile(cfg.output_path);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << cfg.output_path << "\n";
        return 1;
    }

    int dim = getdim(cfg.input_file_path);
    std::cout << "Vector dimension: " << dim << "\n";

    ROCKSDB_NAMESPACE::HNSWGraph hnsw(
        cfg.M, cfg.Mmax, cfg.Ml, cfg.efConstruction,
        &asterDB, outFile, cfg.vector_file_path, dim
    );

    std::cout << "Inserting nodes from " << cfg.input_file_path << std::endl;
    insertFromFile(hnsw, cfg.input_file_path);

    std::cout << "Querying and comparing with ground truth " << cfg.query_file_path << std::endl;
    queryAndCompareWithGroundTruth(hnsw, cfg.query_file_path, cfg.groundtruth_file_path);

    return 0;
}
