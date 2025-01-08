import torch
from Pix2Vox.models.encoder import Encoder
from Pix2Vox.models.decoder import Decoder
from Pix2Vox.models.refiner import Refiner
from Pix2Vox.models.merger import Merger
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for all origins (can be restricted if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify a list of origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods like GET, POST
    allow_headers=["*"],  # Allow all headers
)

class Config:
    def __init__(self):
        self.NETWORK = {
            'TCONV_USE_BIAS': True,
            'LEAKY_VALUE': 0.1
        }
        self.CONST = {
            'N_VOX': 32
        }

cfg = Config()

def load_pix2vox_model(model_path: str, device: str = 'cpu'):
    # Initialize the model components
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Function to remove 'module.' prefix from checkpoint keys if present
    def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    # Modify the state dict if it's wrapped with 'module.'
    encoder_state_dict = remove_module_prefix(checkpoint['encoder_state_dict'])
    decoder_state_dict = remove_module_prefix(checkpoint['decoder_state_dict'])
    
    # Load only the matching keys
    missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state_dict, strict=False)
    print(f"Encoder - Missing keys: {missing_keys}")
    print(f"Encoder - Unexpected keys: {unexpected_keys}")
    
    missing_keys, unexpected_keys = decoder.load_state_dict(decoder_state_dict, strict=False)
    print(f"Decoder - Missing keys: {missing_keys}")
    print(f"Decoder - Unexpected keys: {unexpected_keys}")
    
    if 'refiner_state_dict' in checkpoint:
        refiner_state_dict = remove_module_prefix(checkpoint['refiner_state_dict'])
        missing_keys, unexpected_keys = refiner.load_state_dict(refiner_state_dict, strict=False)
        print(f"Refiner - Missing keys: {missing_keys}")
        print(f"Refiner - Unexpected keys: {unexpected_keys}")
    
    if 'merger_state_dict' in checkpoint:
        merger_state_dict = remove_module_prefix(checkpoint['merger_state_dict'])
        missing_keys, unexpected_keys = merger.load_state_dict(merger_state_dict, strict=False)
        print(f"Merger - Missing keys: {missing_keys}")
        print(f"Merger - Unexpected keys: {unexpected_keys}")

    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    return encoder, decoder, refiner, merger

# Image preprocessing function for handling multiple images
def preprocess_images(image_bytes_list):
    images = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to expected size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for image_bytes in image_bytes_list:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        images.append(image)
    
    # Stack images along the batch dimension (combine all images)
    return torch.cat(images, dim=0)  # Result will be [n_images, 3, 256, 256]


def generate_3d_model(encoder, decoder, refiner, merger, image_bytes_list, n_views=1):
    # Preprocess the input images
    images = preprocess_images(image_bytes_list)
    print(f"Input images shape: {images.shape}")  # [n_images, 3, 256, 256]

    # Reshape the images to have the shape [1, n_views, 3, 256, 256]
    images = images.unsqueeze(0).repeat(1, n_views, 1, 1, 1)  # Repeat for n_views if needed
    print(f"Adjusted image shape (for n_views): {images.shape}")  # [1, n_views, n_images, 256, 256]

    with torch.no_grad():
        # Pass through the encoder
        image_features = encoder(images)  # Shape: [1, n_views, 2048, 2, 2]
        print(f"Image features shape: {image_features.shape}")  # Print image features shape

        # Prepare input for the decoder
        batch_size, n_views, channels, height, width = image_features.shape
        print(f"Features shape: batch_size={batch_size}, n_views={n_views}, feature_map={channels}, height={height}, width={width}")

        # Reshaping the image_features tensor to match the decoder input
        depth = 1  # Set depth to 1 to match the expected shape
        decoder_input = image_features.view(batch_size, n_views, channels, depth, height, width)
        print(f"Decoder input shape (before squeeze): {decoder_input.shape}")

        # Squeeze the depth dimension to match the 5D input required by the decoder
        decoder_input = decoder_input.squeeze(3)  # Remove the depth dimension (since it's 1)
        print(f"Decoder input shape (after squeeze): {decoder_input.shape}")  # Should be [batch_size, n_views, channels, height, width]

        # Now, generate volumes using the decoder
        raw_features, generated_volumes = decoder(decoder_input)  # This should match the expected input shape
        print(f"Generated volumes shape: {generated_volumes.shape}")  # Print generated volumes shape

        # Ensure that the generated volumes tensor is of shape [batch_size, 1, 32, 32, 32]
        if generated_volumes.shape[2] != 32:  # Check if it doesn't match expected size
            generated_volumes = torch.nn.functional.interpolate(
                generated_volumes, size=(32, 32, 32), mode='trilinear', align_corners=False
            )
        print(f"Generated volumes shape after resizing: {generated_volumes.shape}")  # Should now be [1, 1, 32, 32, 32]

        # If the refiner exists, apply the refiner
        if refiner:
            refined_volumes = refiner(generated_volumes)  # Shape: [batch_size, 32, 32, 32]
            print(f"Refined volumes shape: {refined_volumes.shape}")  # Print refined volumes shape

            # Expand the refined volumes to 9 channels by repeating
            refined_volumes = refined_volumes.unsqueeze(1).repeat(1, 9, 1, 1, 1)
            print(f"Refined volumes after expansion: {refined_volumes.shape}")  # Should be [batch_size, 9, 32, 32, 32]

        # Fix the raw_features dimension issue
        print(f"Raw features shape before reshaping: {raw_features.shape}")

        # The key fix: Squeeze the unnecessary dimension (3 in this case)
        raw_features = raw_features.squeeze(1)  # Remove the channel dimension
        print(f"Raw features shape after squeezing: {raw_features.shape}")  # Should be [batch_size, 9, 16, 32, 32]

        # If raw_features has more dimensions than expected, you may need to reshape to match the target shape
        if raw_features.shape[1] != 9:
            # Reshape the raw_features to have the correct shape if necessary
            raw_features = raw_features.view(batch_size, 9, 16, 32, 32)
            print(f"Raw features shape after reshaping: {raw_features.shape}")

        # Merge the final volumes if the merger exists
        if merger:
            # Ensure `raw_features` has the correct shape for the Merger
            print(f"Raw features shape before passing to Merger: {raw_features.shape}")
            print(f"Refined volumes shape before passing to Merger: {refined_volumes.shape}")

            # Pass both tensors to the Merger (they should have matching shapes)
            final_volumes = merger(raw_features, refined_volumes)  # Shape: [batch_size, 32, 32, 32]
            print(f"Final merged volumes shape: {final_volumes.shape}")

        return final_volumes




# Save volumes to a file
def save_volume_to_file(volume_tensor, filename):
    volume = volume_tensor.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
    np.save(filename, volume)  # Save as a .npy file

@app.post("/upload/")
async def upload_images(file: list[UploadFile] = File(...)):
    print("Files received:", len(file))
    model_path = 'models/Pix2Vox-A-ShapeNet.pth'  # Update to correct path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder, refiner, merger = load_pix2vox_model(model_path, device)

    if not file:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    try:
        # Read the file bytes
        image_bytes_list = [await file.read() for file in file]
        print("File bytes read successfully.")
        
        # Generate 3D model from the images
        generated_volumes = generate_3d_model(encoder, decoder, refiner, merger, image_bytes_list)
        print("Model generated successfully.")
        
        # Save the generated model to a .npy file
        volume_filename = f"generated_model.npy"
        save_volume_to_file(generated_volumes, volume_filename)
        print(f"Model saved to {volume_filename}")
        
        # Return the file path or a success message
        return JSONResponse(content={"message": "Files uploaded and model generated successfully!", "model_file": volume_filename})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating model: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
