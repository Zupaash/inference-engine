import os
import torch
import torch.nn as nn
import torch.onnx

# Define a simple architecture matching your engine's operations
class MiniCNN(nn.Module):
    def __init__(self):
        super(MiniCNN, self).__init__()
        # Conv2d: 1 input channel (grayscale), 4 output channels, 3x3 kernel matrix
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=0)
        self.relu = nn.ReLU()
        # MaxPool2d: 2x2 pooling window
        self.pool = nn.MaxPool2d(2, 2)
        # Linear (MatMul): 4 channels * 13 * 13 flattened pixels = 676 inputs -> 10 output classifications
        self.fc = nn.Linear(4 * 13 * 13, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten layer to turn 2D matrix into a 1D vector
        x = self.fc(x)
        return x

def generate_onnx_file(output_path="weights/mnist_model.onnx"):
    # Ensure the target weights/ folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize our network structure
    model = MiniCNN()
    model.eval() # Set to evaluation mode
    
    # Create a dummy image tensor matching standard MNIST size: 1 image, 1 channel, 28x28 pixels
    dummy_input = torch.randn(1, 1, 28, 28)
    
    print(f"Exporting model architecture to {output_path}...")
    
    # Export our structural layout directly to the standard ONNX file format
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True,       # Embed the numerical weights directly inside the file
        opset_version=11,          # Standard, highly compatible configuration version
        do_constant_folding=True,  # Optimizes basic mathematical constants
        input_names=['input'], 
        output_names=['output']
    )
    print("Success! ONNX model file has been generated.")

if __name__ == "__main__":
    generate_onnx_file()
