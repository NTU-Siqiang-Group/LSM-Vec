#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "HNSWUtils.h"
#include "HNSWGraph.h"
#include "Config.h"



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

    lsm_vec::HNSWGraph hnsw(
        config.M, config.Mmax, config.Ml, config.efConstruction, outFile, vectorDim, config
    );

    std::cout << "Inserting nodes from " << config.input_file_path << std::endl;
    insertFromFile(hnsw, config.input_file_path);
    hnsw.printState();

    std::cout << "Querying and comparing with ground truth " << config.query_file_path << std::endl;
    queryAndCompareWithGroundTruth(hnsw, config.query_file_path, config.groundtruth_file_path);

    return 0;
}
