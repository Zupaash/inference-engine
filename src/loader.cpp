#include "loader.hpp"
#include <iostream>
#include <fstream>

// Reads raw binary byte streams directly into a C++ vector of floats
std::vector<float> read_binary_file(const std::string& filepath, size_t total_elements) {
    // 1. Create a memory buffer matching the exact number of elements we expect
    std::vector<float> data(total_elements);
    
    // 2. Open the file pointer in binary mode rather than text mode
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file: " << filepath << std::endl;
        return {}; // Return an empty vector to signal failure
    }
    
    // 3. Calculate how many total physical bytes we need to extract from disk
    size_t total_bytes = total_elements * sizeof(float);
    
    // 4. Force read the block of binary data directly into the sequential array memory
    file.read(reinterpret_cast<char*>(data.data()), total_bytes);
    
    // 5. Safety verification check to see if the file actually matched our sizes
    if (!file) {
        std::cerr << "Error: Only managed to read " << file.gcount() 
                  << " bytes out of " << total_bytes << " from " << filepath << std::endl;
        return {};
    }
    
    return data;
}

// Orchestrator function to systematically load every weight block into our struct bundle
bool load_model(const std::string& directory_path, ModelWeights& weights) {
    std::cout << "Loading model parameters from " << directory_path << "..." << std::endl;
    
    // Core parameters mapping exactly to our Python dimensions:
    // conv1_weight: 4 output * 1 input * 3 kernel * 3 kernel = 36 total floats
    weights.conv1_weight = read_binary_file(directory_path + "/conv1_weight.bin", 36);
    weights.conv1_bias   = read_binary_file(directory_path + "/conv1_bias.bin", 4);
    
    // fc_weight: 10 outputs * 676 flattened features = 6760 total floats
    weights.fc_weight    = read_binary_file(directory_path + "/fc_weight.bin", 6760);
    weights.fc_bias      = read_binary_file(directory_path + "/fc_bias.bin", 10);
    
    // If any vector came back empty, loading failed completely
    if (weights.conv1_weight.empty() || weights.conv1_bias.empty() ||
        weights.fc_weight.empty()    || weights.fc_bias.empty()) {
        std::cerr << "CRITICAL ERROR: Failed to load one or more parameter matrices." << std::endl;
        return false;
    }
    
    std::cout << "Model loaded successfully into execution system memory!" << std::endl;
    return true;
}
