import os
import sys
import argparse
from PIL import Image
import numpy as np
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Import the face enhancer
from face_enhancer import GFPGANFaceEnhancer

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test GFPGAN Face Enhancer")
    parser.add_argument('-i', '--input', type=str, required=True, help='Input image path')
    parser.add_argument('-o', '--output', type=str, default='output', help='Output directory')
    parser.add_argument('-v', '--version', type=str, default='1.4', help='GFPGAN model version')
    parser.add_argument('-s', '--scale', type=int, default=2, help='Upscale factor')
    parser.add_argument('--only_center_face', action='store_true', help='Only process center face')
    parser.add_argument('--bg_upsampler', type=str, default='realesrgan', choices=['realesrgan', 'none'], 
                        help='Background upsampler')
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize face enhancer
    face_enhancer = GFPGANFaceEnhancer()
    
    # Load and process image
    print(f"Processing {args.input}...")
    img = Image.open(args.input).convert('RGB')
    img_np = np.array(img)
    
    # Convert to tensor format (BHWC, float 0-1)
    img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)
    
    # Process the image
    result = face_enhancer.enhance_face(
        img_tensor, 
        version=args.version,
        scale=args.scale,
        only_center_face=args.only_center_face,
        bg_upsampler=args.bg_upsampler,
        output_folder=args.output
    )
    
    print(f"Done! Output saved to {args.output}/restored_imgs/")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 