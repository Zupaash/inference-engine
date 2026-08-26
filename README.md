# A neural-network inference engine from scratch in C++

I will attempt to load a small trained model (export from PyTorch to ONNX), then implement the operators yourself, matmul, conv2d, relu, pooling, and run a real forward pass with no framework. Then optimize: cache-friendly loops, SIMD (AVX/NEON) intrinsics, maybe multithreading. Benchmark against PyTorch.

