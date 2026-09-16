# Custom Neural Inference Engine from Scratch

A zero-dependency, high-performance **Convolutional Neural Network (CNN) Inference Engine** implemented completely from scratch in modern C++ (C++17). 

This project bypasses heavy deep-learning frameworks to load, parse, and execute raw model weights on bare metal. It serves as a lightweight, educational alternative to runtimes like ONNX Runtime or TensorFlow Lite.

---

## 🏗️ Project Architecture & File System

The repository is structured as a hybrid C++ application and offline Python workspace, strictly enforcing a separation of concerns between training-side serialization and execution logic.

```text
inference-engine-scratch/
├── CMakeLists.txt            # Cross-platform build configuration (C++17)
├── .gitignore                # Safely ignores massive model arrays and build builds
├── README.md                 # Project roadmap and documentation
├── tools/                    # Offline tooling workspace
│   ├── requirements.txt      # Python dependencies (torch, onnx, numpy, onnxscript)
│   ├── generate_mnist_onnx.py# Generates target architecture & exports standard ONNX graph
│   └── extract_weights.py    # Shreads composite ONNX files into primitive binary arrays
├── weights/                  # Model data playground (gitignored)
│   ├── mnist_model.onnx      # Pre-trained network checkpoint
│   └── extracted/            # Target folder for binary extractions
│       ├── model_metadata.txt# Structural layout guidelines (dimensions, layer sizes)
│       ├── conv1_weight.bin  # Raw spatial filtering parameters (float32 bytes)
│       └── fc_weight.bin     # Raw dense matrix multiplication parameters
└── src/                      # Core runtime engine source code
    ├── main.cpp              # Entry point orchestrating the forward inference pass
    ├── loader.hpp / .cpp     # High-speed binary weight parsing and memory streamers
    └── layers.hpp / .cpp     # Mathematical implementations of neural network operations
```

---

## 🧮 Supported Mathematical Operators

The engine contains native, hand-written C++ implementations for the core foundational layers required by modern computer vision architectures:

1. **`conv2d` (2D Convolution)**: A sliding-window operator that scans input grids with multi-channel kernel weights and bias offsets to isolate specific geometric features.
2. **`relu` (Rectified Linear Unit)**: A non-linear activation pass that maps over matrices, filtering negative noise values to absolute `0` while maintaining positive feature signals.
3. **`maxpool2d` (Max Pooling)**: A down-sampling window pass (2x2 grid, stride 2) that shrinks matrix dimensions exactly in half, capturing only the most dominant feature spikes.
4. **`fully_connected` (Dense Layer)**: A dot-product matrix multiplication operation that flattens final structural features and maps them against category biases to output categorical class scores.

---

## 🚀 Execution Pipeline & Quick Start

To generate the weight data, configure the build layout, compile, and run a complete forward inference pass, execute the following sequences from your project root:

### 1. Extract the Weights
Set up your python environment and run the shredder script to dump raw parameter matrices:
```bash
pip install -r tools/requirements.txt
python tools/generate_mnist_onnx.py
```

### 3. Execute Inference
Run the compiled runtime binary directly from your terminal console:
# Windows
.\build\Debug\inference_engine.exe

# Mac / Linux
./build/inference_engine
```

---

## 🗺️ Engineering Roadmap & Optimization Targets

- [x] **Phase 1: Project Setup** -> Establish layout blueprints, CMake configuration, and Git protections.
- [x] **Phase 2: Data Extraction** -> Build PyTorch serialization tools and strip weights to raw `.bin` data.
- [x] **Phase 3: C++ Loader** -> Write memory streams using `reinterpret_cast` to move bytes to RAM.
- [x] **Phase 4: Math Pipeline** -> Implement custom nested-loop mathematical layers (`conv2d`, `relu`, `pool`, `matmul`).
- [x] **Phase 5: Forward Pass Integration** -> Wire up a sequential execution pass and evaluate inputs.
- [ ] **Phase 6: Real Image Parsing** -> Integrate low-profile image decoders (`stb_image`) to test real pixel arrays.
- [ ] **Phase 7: Hardware Optimization** -> Speed up matrix math using **Cache-friendly layout flattening**, **SIMD (AVX2/NEON) vector intrinsics**, and multi-core **OpenMP multithreading**.
```bash

### 2. Compile the Engine
cmake -B build
cmake --build build
Leverage CMake to configure and compile the optimized C++ executable:
```bash
cmake -B build
cmake --build build
```

### 3. Execute Inference
Run the compiled runtime binary directly from your terminal console:
```bash
# Windows
.\build\Debug\inference_engine.exe

# Mac / Linux
./build/inference_engine
```

---

## 🗺️ Next Steps & Optimization Targets

- **Real Image Parsing**: Integrate low-profile image decoders (`stb_image`) to test real pixel arrays.
- **Hardware Optimization**: Speed up matrix math using **Cache-friendly layout flattening**, **SIMD (AVX2/NEON) vector intrinsics**, and multi-core **OpenMP multithreading**.
