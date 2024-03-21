from api_image_processor import image_utility # Import Essential Classes and Packages
import faiss
from FAISS_api_search import Search
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models
import uvicorn

# Set environment variable to avoid duplicate library loading issues (Remove it if already configured on the System!)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Initialize image utility and search classes
img_util = image_utility()
srch = Search()

# Define a custom feature extraction model
class FeatureExtractor(nn.Module):

    def __init__(self, decoder: dict = None):

        super(FeatureExtractor, self).__init__()
        # Load pre-trained ResNet50 model
        resnet_model_initial = models.resnet50(pretrained=True)
        
        # Freeze all layers except the last two
        for name, param in resnet_model_initial.named_parameters():
            if 'layers.4.2' in name or 'layers.4.1' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        added_layers = nn.Sequential(nn.ReLU(), nn.Linear(1000, 14))
        self.resnet_model = nn.Sequential(resnet_model_initial, added_layers)
        self.decoder = decoder

    def forward(self, image):
        x = self.resnet_model(image)
        return x

    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

# Attempt to load the feature extraction model and FAISS index
try:

    feature_model = FeatureExtractor()
    feature_model.load_state_dict(torch.load('image_model.pt'))
    feature_model.eval()
    pass

except:

    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:

    index = faiss.read_index('appended_file.pkl') # Load the pickle file
    pass

except:
    
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

# Create a FastAPI instance
app = FastAPI()
print("Starting server")

# Health check endpoint to verify server status
@app.get('/healthcheck')

def healthcheck():
  
  msg = "API is up and running!"
  
  return {"message": msg}

# Endpoint to predict image embeddings  
@app.post('/predict/feature_embedding')

def predict_image(image: UploadFile = File(...)):

    pil_image = Image.open(image.file)
    img_tfrm = img_util.image_transform(pil_image)
    img_tfrm = img_tfrm.unsqueeze(0)
    img_emb = feature_model(img_tfrm)
    return JSONResponse(content={"features": img_emb.tolist()[0]}) # Return the image embeddings
    
# Endpoint to predict similar images using FAISS index          
@app.post('/predict/similar_images')

def predict_combined(image: UploadFile = File(...)):
    
    pil_image = Image.open(image.file) # Opening Image File
    img_tfrm = img_util.image_transform(pil_image) # Transforming Image to a Tensor
    img_tfrm = img_tfrm.unsqueeze(0) # Converting 3D Tensor to 4D Tensor to match the Dimensions for the Model Input
    img_emb = feature_model(img_tfrm)
    s_index = srch.search_img(img_emb) # Calling Search Function to find Similar Images 
    
    return JSONResponse(content={"similar_index": s_index,}) # Return the index of similar images

# Run the FastAPI server    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)