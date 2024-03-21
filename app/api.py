import uvicorn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from fastapi import File
from fastapi import UploadFile
import faiss
import torch
import torch.nn as nn
from FAISS_api_search import Search
from torchvision import  models
##############################################################
# TODO                                                       #
# Import your image processing script here                 #
##############################################################
from api_image_processor import image_utility
img_util = image_utility()
srch = Search()

class FeatureExtractor(nn.Module):
    def __init__(self, decoder: dict = None):
        super(FeatureExtractor, self).__init__()

##############################################################
# TODO                                                       #
# Populate the __init__ method, so that it contains the same #
# structure as the model you used to train the image model   #
##############################################################
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
'''
# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str
'''


try:
#################################################################
# TODO                                                          #
# Load the Feature Extraction model. Above, we have initialized #
# a class that inherits from nn.Module, and has the same        #
# structure as the model that you used for training it. Load    #
# the weights in it here.                                       #
#################################################################
    feature_model = FeatureExtractor()
    feature_model.load_state_dict(torch.load('image_model.pt'))
    feature_model.eval()
    pass
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:
##################################################################
# TODO                                                           #
# Load the FAISS model. Use this space to load the FAISS model   #
# which is was saved as a pickle with all the image embeddings   #
# fit into it.                                                   #
##################################################################
    index = faiss.read_index('appended_file.pkl')
    pass
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")


app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

  
@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    pil_image = Image.open(image.file)
    
    ################################################################
    # TODO                                                         #
    # Process the input and use it as input for the feature        #
    # extraction model image. File is the image that the user      #
    # sent to your API. Apply the corresponding methods to extract #
    # the image features/embeddings.                               #
    ################################################################
    img_tfrm = img_util.image_transform(pil_image)
    img_tfrm = img_tfrm.unsqueeze(0)
    img_emb = feature_model(img_tfrm)
    return JSONResponse(content={"features": img_emb.tolist()[0]}) # Return the image embeddings here
    
        
  
@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...)):
    
    pil_image = Image.open(image.file)
    img_tfrm = img_util.image_transform(pil_image)
    img_tfrm = img_tfrm.unsqueeze(0)
    img_emb = feature_model(img_tfrm)
    s_index = srch.search_img(img_emb)
    #####################################################################
    # TODO                                                              #
    # Process the input  and use it as input for the feature            #
    # extraction model.File is the image that the user sent to your API #   
    # Once you have feature embeddings from the model, use that to get  # 
    # similar images by passing the feature embeddings into FAISS       #
    # model. This will give you index of similar images.                #            
    #####################################################################

    return JSONResponse(content={"similar_index": s_index,}) # Return the index of similar images here, pass a dict here
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)