import os
import struct
import onnx
from onnx import numpy_helper

def extract_onnx_weights(onnx_model_path, output_dir="weights/extracted"):
    # 1. Open the model graph
    model = onnx.load(onnx_model_path)
    graph = model.graph
    
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "model_metadata.txt")
    
    with open(metadata_path, "w") as meta_file:
        print(f"Extracting weights from {onnx_model_path}...")
        
        # 2. Iterate over all trained layers (weights and biases)
        for initializer in graph.initializer:
            name = initializer.name
            # Sanitize name to make it safe for file paths
            safe_name = name.replace("/", "_").replace(".", "_")
            
            # 3. Convert raw bytes to a clean NumPy array
            weight_array = numpy_helper.to_array(initializer)
            
            # 4. Force everything to float32 to match standard C++ 'float'
            if weight_array.dtype != "float32":
                weight_array = weight_array.astype("float32")
                
            shape = weight_array.shape
            shape_str = ",".join(map(str, shape))
            
            # 5. Log structural metadata
            meta_file.write(f"{safe_name} [{shape_str}] {weight_array.size}\n")
            
            # 6. Save raw sequential bytes to disk
            bin_path = os.path.join(output_dir, f"{safe_name}.bin")
            with open(bin_path, "wb") as bin_file:
                bin_file.write(struct.pack(f"{weight_array.size}f", *weight_array.flatten()))
                
            print(f" -> Saved: {safe_name}.bin | Shape: ({shape_str})")

    print(f"\nSuccess! All weights exported to folder: '{output_dir}'")

# Execute extraction on our target model
if __name__ == "__main__":
    extract_onnx_weights("weights/mnist_model.onnx")
