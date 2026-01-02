#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "utils.h"
#include "lsm_vec_db.h"
#include "lsm_vec_index.h"
#include "config.h"



int main(int argc, char* argv[])
{
    Config config = Config::Parse(argc, argv);

    std::ofstream outFile(config.output_path);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << config.output_path << "\n";
        return 1;
    }

    int vectorDim = getdim(config.input_file_path);
    std::cout << "Vector dimension: " << vectorDim << "\n";

    lsm_vec::LSMVecDBOptions options;
    options.dim = vectorDim;
    options.m = config.M;
    options.m_max = config.Mmax;
    options.m_level = config.Ml;
    options.ef_construction = config.efConstruction;
    options.vec_file_capacity = config.vec_file_capacity;
    options.paged_max_cached_pages = config.paged_max_cached_pages;
    options.vector_storage_type = config.vector_storage_type;
    options.db_target_size = config.db_target_size;
    options.random_seed = config.random_seed;
    options.vector_file_path = config.vector_file_path;

    lsm_vec::LSMVec hnsw(config.db_path, options, outFile);

    std::cout << "Inserting nodes from " << config.input_file_path << std::endl;
    insertFromFile(hnsw, config.input_file_path);
    hnsw.printState();

    std::cout << "Querying and comparing with ground truth " << config.query_file_path << std::endl;
    queryAndCompareWithGroundTruth(hnsw, config.query_file_path, config.groundtruth_file_path);

    return 0;
}
