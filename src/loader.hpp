#ifndef LOADER_HPP
#define LOADER_HPP

#include <string>
#include <vector>

struct ModelWeights {
    std::vector<float> conv1_weight;
    std::vector<float> conv1_bias;
    std::vector<float> fc_weight;
    std::vector<float> fc_bias;
};

std::vector<float> read_binary_file(const std::string& filepath, size_t total_elements);
bool load_model(const std::string& directory_path, ModelWeights& weights);

#endif // LOADER_HPP
