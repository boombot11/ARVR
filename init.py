from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
import numpy as np
from Pix2Vox.models.encoder import Encoder
from Pix2Vox.models.decoder import Decoder
from Pix2Vox.models.refiner import Refiner
from Pix2Vox.models.merger import Merger
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI()

# Allow CORS for all origins (can be restricted if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify a list of origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods like GET, POST
    allow_headers=["*"],  # Allow all headers
)

# Define your model configuration
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

# Initialize Pix2Vox model components
def load_pix2vox_model(model_path: str, device: str = 'cpu'):
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)

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

    encoder_state_dict = remove_module_prefix(checkpoint['encoder_state_dict'])
    decoder_state_dict = remove_module_prefix(checkpoint['decoder_state_dict'])
    
    encoder.load_state_dict(encoder_state_dict, strict=False)
    decoder.load_state_dict(decoder_state_dict, strict=False)

    if 'refiner_state_dict' in checkpoint:
        refiner_state_dict = remove_module_prefix(checkpoint['refiner_state_dict'])
        refiner.load_state_dict(refiner_state_dict, strict=False)
    
    if 'merger_state_dict' in checkpoint:
        merger_state_dict = remove_module_prefix(checkpoint['merger_state_dict'])
        merger.load_state_dict(merger_state_dict, strict=False)

    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    return encoder, decoder, refiner, merger

# Image preprocessing function
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to expected size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Generate 3D model
def generate_3d_model(encoder, decoder, refiner, merger, image_bytes, n_views=1):
    image = preprocess_image(image_bytes)

    # Reshape the image to have the shape [1, n_views, 3, 256, 256]
    image = image.unsqueeze(0).repeat(1, n_views, 1, 1, 1)

    with torch.no_grad():
        image_features = encoder(image)

        batch_size, n_views, channels, height, width = image_features.shape

        decoder_input = image_features.view(batch_size, n_views, channels, 1, height, width)
        decoder_input = decoder_input.squeeze(3)

        raw_features, generated_volumes = decoder(decoder_input)

        if generated_volumes.shape[2] != 32:
            generated_volumes = torch.nn.functional.interpolate(
                generated_volumes, size=(32, 32, 32), mode='trilinear', align_corners=False
            )

        if refiner:
            refined_volumes = refiner(generated_volumes)
            refined_volumes = refined_volumes.unsqueeze(1).repeat(1, 9, 1, 1, 1)

        if merger:
            raw_features = raw_features.squeeze(1)
            raw_features = torch.nn.functional.interpolate(
                raw_features, size=(32, 32, 32), mode='trilinear', align_corners=False
            )

            final_volumes = merger(raw_features, refined_volumes)
            return final_volumes

        return generated_volumes

# Save volumes to a file in the 'generated' folder
def save_volume_to_file(volume_tensor, filename):
    volume = volume_tensor.cpu().numpy()  # Remove batch dimension and move to CPU
    save_path = os.path.join('generated', filename)
    np.save(save_path, volume)
    return save_path

# POST endpoint to handle image upload and generate 3D model
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    model_path = 'models/Pix2Vox-A-ShapeNet.pth'  # Update to correct path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder, refiner, merger = load_pix2vox_model(model_path, device)
    
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    try:
        # Read the file bytes
        image_bytes = await file.read()

        # Generate 3D model from the image
        generated_volumes = generate_3d_model(encoder, decoder, refiner, merger, image_bytes)

        # Save the generated model to a .npy file in the 'generated' folder
        volume_filename = f"generated_model_{file.filename}.npy"
        volume_filename="3dmodel.npy"
        file_path = save_volume_to_file(generated_volumes, volume_filename)
        # npy_data = np.load('./generated/3dmodel.npy')

        # # Print the shape of the data
        # print(f"Shape of the data: {npy_data.shape}")

        # # Check the data type to ensure it's correct
        # print(f"Data type: {npy_data.dtype}")

        # # Optionally, check some of the values to make sure it's not empty
        # print(f"Some of the values: {npy_data[:5]}")  # Print first 5 values
        # Return the file path to the frontend
        return JSONResponse(content={"message": "File uploaded and model generated successfully!", "model_file": f"/generated/3dmodel.npy"})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating model: {str(e)}")

# GET endpoint to retrieve the .npy model file for rendering
@app.get("/generated/{filename}")
async def get_generated_model(filename: str):
    file_path = os.path.join('generated', filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

# Run the FastAPI app using Uvicorn
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
