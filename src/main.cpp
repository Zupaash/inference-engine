
#include <iostream>
#include "loader.hpp"

int main() {
    std::ios_base::sync_with_stdio(false);
    
    std::cout << "=== Custom Neural Inference Engine ===" << std::endl;
    
    // Allocate our blank structural vault to hold the tensors
    ModelWeights weights;
    
    // Define the path to where your extracted weights live
    std::string model_path = "weights/extracted";
    
    // Execute the memory streaming pass
    if (!load_model(model_path, weights)) {
        return 1; // Terminate with error code 1 if files are missing or broken
    }
    
    // Print verification logs to prove the arrays exist in memory
    std::cout << "\n--- Memory Validation Log ---" << std::endl;
    std::cout << "Conv1 Weights Vector Size: " << weights.conv1_weight.size() << " floats" << std::endl;
    std::cout << "FC Weights Vector Size:    " << weights.fc_weight.size() << " floats" << std::endl;
    std::cout << "\nReady for Phase 3 mathematical operations!" << std::endl;
    
    return 0;
}



