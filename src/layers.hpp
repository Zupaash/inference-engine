#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <vector>

// 1. 2D Convolution Operation
void conv2d(const std::vector<float>& input, int in_c, int in_h, int in_w,
            const std::vector<float>& weights, const std::vector<float>& bias,
            int out_c, int k_h, int k_w, std::vector<float>& output);

// 2. ReLU Activation Operation
void relu(std::vector<float>& tensor);

// 3. Max Pooling 2D Operation (2x2 stride)
void maxpool2d(const std::vector<float>& input, int c, int in_h, int in_w,
               std::vector<float>& output);

// 4. Fully Connected Layer (Matrix Multiplication + Bias)
void fully_connected(const std::vector<float>& input, const std::vector<float>& weights,
                     const std::vector<float>& bias, int in_features, int out_features,
                     std::vector<float>& output);

#endif // LAYERS_HPP
