import os
import cv2
import torch
import numpy as np
import glob
from PIL import Image
import folder_paths
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
import torchvision.transforms as transforms

# Check for GFPGAN
try:
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
    GFPGAN_AVAILABLE = True
except ImportError:
    print("GFPGAN not available, downloading from source...")
    GFPGAN_AVAILABLE = False
    import sys
    import subprocess
    
    # Clone GFPGAN if not available
    subprocess.run(["git", "clone", "https://github.com/TencentARC/GFPGAN.git", "gfpgan_repo"], check=True)
    sys.path.append("gfpgan_repo")
    try:
        from gfpgan_repo.gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
        GFPGAN_AVAILABLE = True
    except ImportError:
        print("Failed to import GFPGAN, please install it manually")

# Define the GFPGAN Face Enhancer node
class GFPGANFaceEnhancer:
    def __init__(self):
        self.model = None
        self.face_helper = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "GFPGANv1.4.pth")
        
        # Create the models directory if it doesn't exist
        os.makedirs(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models"), exist_ok=True)
        
        # Check if model exists, if not download it
        if not os.path.exists(self.model_path):
            print(f"Downloading GFPGAN model to {self.model_path}...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                self.model_path
            )
        
    def load_model(self, version='1.4'):
        if self.model is not None:
            return
        
        # Initialize the face helper
        self.face_helper = FaceRestoreHelper(
            upscale_factor=2,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device
        )
        
        # Initialize the GFPGAN model
        if version == '1.4':
            self.model = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=2,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True
            )
            
            # Load the model weights
            checkpoint = torch.load(self.model_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint['params_ema'], strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            print("GFPGAN model loaded")
        else:
            raise ValueError(f"Unsupported GFPGAN version: {version}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "version": (["1.4"],),
                "scale": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "only_center_face": ("BOOLEAN", {"default": False}),
                "bg_upsampler": (["realesrgan", "none"], {"default": "realesrgan"}),
            },
            "optional": {
                "output_folder": ("STRING", {"default": "gfpgan_output"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance_face"
    CATEGORY = "image/restoration"

    def enhance_face(self, image, version='1.4', scale=2, only_center_face=False, bg_upsampler="realesrgan", output_folder="gfpgan_output"):
        # Load the model if not loaded
        self.load_model(version)
        
        # Ensure output directory exists
        output_dir = os.path.join(folder_paths.get_output_directory(), output_folder)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "restored_imgs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "restored_faces"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "cropped_faces"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "cmp"), exist_ok=True)
        
        # Process each image in the batch
        enhanced_images = []
        batch_size = image.shape[0]
        
        for i in range(batch_size):
            # Convert from tensor to numpy image (ComfyUI uses BHWC format with float values 0-1)
            # Convert to uint8 in RGB format for processing
            img = (image[i].cpu().numpy() * 255).astype(np.uint8)
            
            # Process image with GFPGAN
            enhanced_img = self.process_single_image(img, scale, only_center_face, bg_upsampler)
            
            # Convert back to tensor format for ComfyUI
            enhanced_tensor = torch.from_numpy(enhanced_img.astype(np.float32) / 255.0)
            enhanced_images.append(enhanced_tensor)
            
            # Save the result
            result_img = Image.fromarray(enhanced_img)
            result_img.save(os.path.join(output_dir, "restored_imgs", f"enhanced_{i:05d}.png"))
        
        # Stack the enhanced images into a batch tensor
        return (torch.stack(enhanced_images, dim=0),)
    
    def process_single_image(self, img_np, scale, only_center_face, bg_upsampler):
        # BGR to RGB for facexlib
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Detect and extract faces
        self.face_helper.clean_all()
        self.face_helper.read_image(img_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=only_center_face)
        self.face_helper.align_warp_face()
        
        # Face restoration
        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            # Prepare input
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            cropped_face_t = normalize(cropped_face_t).unsqueeze(0).to(self.device)
            
            try:
                with torch.no_grad():
                    output = self.model(cropped_face_t)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                self.face_helper.add_restored_face(restored_face)
            except Exception as e:
                print(f'Error in enhancing face: {e}')
                # If error, just use the original face
                self.face_helper.add_restored_face(cropped_face)
        
        # Paste faces back
        if bg_upsampler == "realesrgan" and scale > 1:
            try:
                # Import RealESRGAN for background upsampling
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                bg_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                bg_upsampler_model = RealESRGANer(
                    scale=2,
                    model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                    dni_weight=None,
                    model=bg_model,
                    tile=400,
                    tile_pad=10,
                    pre_pad=0,
                    half=False,
                    device=self.device
                )
                
                # Upscale background with Real-ESRGAN
                bg_img = bg_upsampler_model.enhance(img_bgr, outscale=scale)[0]
            except ImportError:
                print("RealESRGAN not found, using bicubic upsampling for background")
                bg_img = cv2.resize(img_bgr, (img_bgr.shape[1] * scale, img_bgr.shape[0] * scale), 
                                    interpolation=cv2.INTER_CUBIC)
        else:
            # Use bicubic upsampling for background
            if scale > 1:
                bg_img = cv2.resize(img_bgr, (img_bgr.shape[1] * scale, img_bgr.shape[0] * scale), 
                                   interpolation=cv2.INTER_CUBIC)
            else:
                bg_img = img_bgr
        
        # Paste enhanced faces to background
        self.face_helper.get_inverse_affine(None)
        restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
        
        # Convert back to RGB for output
        restored_img_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        
        return restored_img_rgb


# Define batch processing node for folder input
class GFPGANFolderProcessor:
    def __init__(self):
        self.face_enhancer = GFPGANFaceEnhancer()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_folder": ("STRING", {"default": ""}),
                "version": (["1.4"],),
                "scale": ("INT", {"default": 2, "min": 1, "max": 4, "step": 1}),
                "only_center_face": ("BOOLEAN", {"default": False}),
                "bg_upsampler": (["realesrgan", "none"], {"default": "realesrgan"}),
                "output_folder": ("STRING", {"default": "gfpgan_output"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_folder"
    CATEGORY = "image/restoration"

    def process_folder(self, image_folder, version, scale, only_center_face, bg_upsampler, output_folder):
        # Check if folder exists
        if not os.path.exists(image_folder):
            raise ValueError(f"Folder not found: {image_folder}")
        
        # Get image files from the folder
        image_files = []
        for ext in ['png', 'jpg', 'jpeg', 'webp', 'bmp']:
            image_files.extend(glob.glob(os.path.join(image_folder, f"*.{ext}")))
            image_files.extend(glob.glob(os.path.join(image_folder, f"*.{ext.upper()}")))
        
        if not image_files:
            raise ValueError(f"No image files found in folder: {image_folder}")
        
        # Sort files to process them in order
        image_files.sort()
        
        # Process each image
        enhanced_images = []
        for i, img_path in enumerate(image_files):
            try:
                # Read image
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img)
                
                # Convert to tensor format for ComfyUI
                img_tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0)
                
                # Process with GFPGAN (directly using the underlying process function)
                self.face_enhancer.load_model(version)
                
                # BGR to RGB for facexlib
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Process with face helper
                self.face_enhancer.face_helper.clean_all()
                self.face_enhancer.face_helper.read_image(img_bgr)
                self.face_enhancer.face_helper.get_face_landmarks_5(only_center_face=only_center_face)
                self.face_enhancer.face_helper.align_warp_face()
                
                # Get enhanced image
                enhanced_img = self.face_enhancer.process_single_image(img_np, scale, only_center_face, bg_upsampler)
                
                # Convert to tensor
                enhanced_tensor = torch.from_numpy(enhanced_img.astype(np.float32) / 255.0)
                enhanced_images.append(enhanced_tensor)
                
                # Save the result
                output_dir = os.path.join(folder_paths.get_output_directory(), output_folder)
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(os.path.join(output_dir, "restored_imgs"), exist_ok=True)
                
                base_filename = os.path.basename(img_path)
                result_img = Image.fromarray(enhanced_img)
                result_img.save(os.path.join(output_dir, "restored_imgs", f"enhanced_{i:05d}_{base_filename}"))
                
                print(f"Processed {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Stack all processed images into a batch
        if enhanced_images:
            return (torch.stack(enhanced_images, dim=0),)
        else:
            # Return empty tensor if no images processed
            return (torch.zeros((0, 3, 512, 512), dtype=torch.float32),)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GFPGANFaceEnhancer": GFPGANFaceEnhancer,
    "GFPGANFolderProcessor": GFPGANFolderProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GFPGANFaceEnhancer": "GFPGAN Face Enhancer",
    "GFPGANFolderProcessor": "GFPGAN Folder Processor"
} 