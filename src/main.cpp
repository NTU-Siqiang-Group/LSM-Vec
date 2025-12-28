#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "HNSWUtils.h"
#include "HNSWGraph.h"
#include "Config.h"



int main(int argc, char* argv[])
{
    Config cfg_ = Config::Parse(argc, argv);

    std::ofstream outFile(cfg_.output_path);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << cfg_.output_path << "\n";
        return 1;
    }

    int dim = getdim(cfg_.input_file_path);
    std::cout << "Vector dimension: " << dim << "\n";

    lsm_vec::HNSWGraph hnsw(
        cfg_.M, cfg_.Mmax, cfg_.Ml, cfg_.efConstruction, outFile, cfg_.vector_file_path, dim, cfg_
    );

    std::cout << "Inserting nodes from " << cfg_.input_file_path << std::endl;
    insertFromFile(hnsw, cfg_.input_file_path);

    std::cout << "Querying and comparing with ground truth " << cfg_.query_file_path << std::endl;
    queryAndCompareWithGroundTruth(hnsw, cfg_.query_file_path, cfg_.groundtruth_file_path);

    return 0;
}
