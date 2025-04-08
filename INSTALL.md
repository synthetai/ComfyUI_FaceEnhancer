# ComfyUI_FaceEnhancer Installation Guide

This guide provides detailed instructions for installing and using the ComfyUI_FaceEnhancer custom node.

## Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended)
- ComfyUI already installed and working

## Step 1: Clone the Repository

Clone this repository into your ComfyUI's custom_nodes directory:

```bash
cd /path/to/ComfyUI/custom_nodes/
git clone https://github.com/YourUsername/ComfyUI_FaceEnhancer.git
cd ComfyUI_FaceEnhancer
```

## Step 2: Install Dependencies

Install the required dependencies:

```bash
pip install basicsr basicsr-fixed facexlib realesrgan
pip install -r requirements.txt
```

If you're using a specific CUDA version, you may want to install PyTorch with the appropriate CUDA version:

```bash
pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

## Step 3: Download Pre-trained Model

Download the GFPGAN pre-trained model:

```bash
mkdir -p models
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P models/
```

Note: If wget is not available on your system, you can manually download the model from the URL and place it in the `models` directory.

## Step 4: Restart ComfyUI

If ComfyUI is already running, restart it to load the new custom node.

## Usage

### Single Image Processing

1. Use the "Load Image" node to load your input image
2. Connect it to the "GFPGAN Face Enhancer" node
3. Configure the parameters:
   - version: GFPGAN model version (1.4)
   - scale: Upscaling factor (1-4)
   - only_center_face: If true, only enhances the center face in the image
   - bg_upsampler: Background upsampling method
   - output_folder: Directory to save processed images
4. Connect the output to a "Save Image" node

### Video Processing

1. Use the "Load Video" node to load your input video
2. Connect it to the "GFPGAN Face Enhancer" node
3. Configure the parameters as needed
4. Connect the output to a "Video Combine" node to create the enhanced video

## Output Directories

The node creates the following subdirectories in your specified output folder:

- `restored_imgs`: The final enhanced images
- `restored_faces`: Only the enhanced faces
- `cropped_faces`: The original cropped faces
- `cmp`: Comparison images showing before and after

## Troubleshooting

- If you encounter CUDA errors, make sure you have the correct PyTorch version installed for your CUDA version
- If the model fails to download automatically, download it manually and place it in the `models` directory
- For memory issues, try processing images at a smaller scale
- For more complex issues, check the official [GFPGAN repository](https://github.com/TencentARC/GFPGAN) 