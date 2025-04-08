# ComfyUI_FaceEnhancer

A ComfyUI custom node for enhancing faces in images and videos using GFPGAN.

## Installation

1. Clone this repository into ComfyUI's custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YourUsername/ComfyUI_FaceEnhancer.git
cd ComfyUI_FaceEnhancer
```

2. Install the required dependencies:
```bash
pip install basicsr basicsr-fixed facexlib realesrgan
pip install -r requirements.txt
```

3. Download the pre-trained GFPGAN model:
```bash
mkdir -p models
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P models/
```

## Usage

1. In ComfyUI, you'll find a new node called "GFPGAN Face Enhancer"
2. Connect an image or image batch to enhance faces
3. Configure the options and run the workflow
4. The output will be enhanced images with improved facial details

## Options

- **image**: Input single image
- **image_folder**: Input folder containing multiple images
- **output_folder**: Folder to save results (will create subdirectories)
- **version**: GFPGAN model version (1, 1.2, 1.3, 1.4)
- **scale**: Upscaling factor
- **only_center_face**: Only process the center face
- **bg_upsampler**: Background upsampler method

## Credits

This node is based on the [GFPGAN project](https://github.com/TencentARC/GFPGAN) by TencentARC.
