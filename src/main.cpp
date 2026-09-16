#include <iostream>
#include <vector>
#include <numeric>   // Gives us std::distance
#include <algorithm> // Gives us std::max_element
#include "loader.hpp"
#include "layers.hpp"

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cout << "=== Custom Neural Inference Engine ===" << std::endl;
    
    // 1. Allocate space and load our real weight files into memory
    ModelWeights weights;
    if (!load_model("weights/extracted", weights)) {
        return 1;
    }
    
    // 2. Create a dummy test image matching MNIST size (1 channel, 28x28 pixels = 784 total floats)
    // We will initialize it with some arbitrary test pixel scores (e.g., 0.5f)
    std::cout << "\nCreating a 28x28 dummy input image..." << std::endl;
    std::vector<float> input_image(784, 0.5f);
    
    // 3. Allocate intermediate memory buffers to hold results between layers
    // Conv Output: 4 channels * 26 height * 26 width = 2704 elements
    std::vector<float> conv_output(4 * 26 * 26);
    
    // MaxPool Output: 4 channels * 13 height * 13 width = 676 elements
    std::vector<float> pool_output(4 * 13 * 13);
    
    // Final Class Output: 10 target digit categories
    std::vector<float> final_scores(10);
    
    // 4. Execute the Forward Inference Pipeline Pass
    std::cout << "Running forward pass..." << std::endl;
    
    // Step A: Run Convolution (28x28 image -> 26x26 output maps)
    conv2d(input_image, 1, 28, 28, weights.conv1_weight, weights.conv1_bias, 4, 3, 3, conv_output);
    
    // Step B: Run ReLU Activation (wipes away any negative noise scores)
    relu(conv_output);
    
    // Step C: Run Max Pooling (shrinks 26x26 maps down in half to 13x13 features)
    maxpool2d(conv_output, 4, 26, 26, pool_output);
    
    // Step D: Run Fully Connected Layer (calculates voting scores for digits 0-9)
    fully_connected(pool_output, weights.fc_weight, weights.fc_bias, 676, 10, final_scores);
    
    // 5. Read the final predictions out to the console screen
    std::cout << "\n--- Classification Output Scores ---" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Digit [" << i << "]: " << final_scores[i] << std::endl;
    }
    
    // Find the highest score index using a built-in iterator locator
    auto max_it = std::max_element(final_scores.begin(), final_scores.end());
    int predicted_digit = std::distance(final_scores.begin(), max_it);
    
    std::cout << "\n*** ENGINE PREDICTION: The image is a " << predicted_digit << "! ***" << std::endl;
    
    return 0;
}




