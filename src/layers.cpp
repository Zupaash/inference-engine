#include "layers.hpp"
#include <algorithm> // Gives us std::max

// 1. 2D Convolution implementation using nested sliding windows
void conv2d(const std::vector<float>& input, int in_c, int in_h, int in_w,
            const std::vector<float>& weights, const std::vector<float>& bias,
            int out_c, int k_h, int k_w, std::vector<float>& output) {
    
    // Calculate the dimensions of our output spatial grid
    int out_h = in_h - k_h + 1;
    int out_w = in_w - k_w + 1;
    
    // Loop 1: Iterate over every output channel (our 4 feature maps)
    for (int oc = 0; oc < out_c; ++oc) {
        // Loop 2 & 3: Slide our filter vertically and horizontally over the output grid row/col
        for (int r = 0; r < out_h; ++r) {
            for (int c = 0; c < out_w; ++c) {
                
                // Start with the specific bias score for this output channel
                float sum = bias[oc];
                
                // Loop 4: Iterate over incoming input channels (MNIST grayscale is just 1)
                for (int ic = 0; ic < in_c; ++ic) {
                    // Loop 5 & 6: Multiply the 3x3 kernel matrix against the current image neighborhood
                    for (int kr = 0; kr < k_h; ++kr) {
                        for (int kc = 0; kc < k_w; ++kc) {
                            
                            // Map 2D coordinates into our flat 1D input array index
                            int input_r = r + kr;
                            int input_c = c + kc;
                            int input_idx = (ic * in_h * in_w) + (input_r * in_w) + input_c;
                            
                            // Map 4D kernel matrix coordinates into our flat 1D weight array index
                            int weight_idx = (oc * in_c * k_h * k_w) + (ic * k_h * k_w) + (kr * k_w) + kc;
                            
                            sum += input[input_idx] * weights[weight_idx];
                        }
                    }
                }
                
                // Write the final accumulated score to our flat output matrix
                int out_idx = (oc * out_h * out_w) + (r * out_w) + c;
                output[out_idx] = sum;
            }
        }
    }
}

// 2. ReLU Activation implementation
void relu(std::vector<float>& tensor) {
    // A simple, single-loop pass over every floating point block in memory
    for (size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = std::max(0.0f, tensor[i]);
    }
}

// 3. Max Pooling 2D implementation (2x2 filter window, stride of 2)
void maxpool2d(const std::vector<float>& input, int c, int in_h, int in_w,
               std::vector<float>& output) {
    
    // Calculate our new halved output spatial dimensions
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    
    // Loop 1: Iterate over our 4 feature channels independently
    for (int ch = 0; ch < c; ++ch) {
        // Loop 2 & 3: Move across rows and columns of our new shrunk output grid
        for (int r = 0; r < out_h; ++r) {
            for (int c_out = 0; c_out < out_w; ++c_out) {
                
                // Map the output coordinates back to the original top-left corner of our 2x2 input block
                int input_r_start = r * 2;
                int input_c_start = c_out * 2;
                
                // Initialize our max tracking variable to the very first value in the 2x2 window
                int first_idx = (ch * in_h * in_w) + (input_r_start * in_w) + input_c_start;
                float max_val = input[first_idx];
                
                // Loop 4 & 5: Check all 4 spots inside the small 2x2 grid window
                for (int kr = 0; kr < 2; ++kr) {
                    for (int kc = 0; kc < 2; ++kc) {
                        int curr_r = input_r_start + kr;
                        int curr_c = input_c_start + kc;
                        int curr_idx = (ch * in_h * in_w) + (curr_r * in_w) + curr_c;
                        
                        // If we find a higher number, update our tracker
                        max_val = std::max(max_val, input[curr_idx]);
                    }
                }
                
                // Write the winning maximum score to our clean output vector array
                int out_idx = (ch * out_h * out_w) + (r * out_w) + c_out;
                output[out_idx] = max_val;
            }
        }
    }
}

// 4. Fully Connected Layer (Dense Matrix Multiplication + Bias)
void fully_connected(const std::vector<float>& input, const std::vector<float>& weights,
                     const std::vector<float>& bias, int in_features, int out_features,
                     std::vector<float>& output) {
    
    // Loop 1: Iterate over each output class (our 10 digit categories)
    for (int o = 0; o < out_features; ++o) {
        
        // Start our calculation with the baseline bias value for this specific digit class
        float total_score = bias[o];
        
        // Loop 2: Perform a dot-product multiplication against all 676 incoming features
        for (int i = 0; i < in_features; ++i) {
            
            // Map the 2D row/column weight index into our flat 1D memory row array
            int weight_idx = (o * in_features) + i;
            
            // Multiply the input clue by the judge's weight modifier and accumulate the sum
            total_score += input[i] * weights[weight_idx];
        }
        
        // Save the judge's final raw classification score into our output vector slots
        output[o] = total_score;
    }
}
