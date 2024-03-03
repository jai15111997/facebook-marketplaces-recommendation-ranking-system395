import torch
from PIL import Image
import torchvision.transforms as transforms

def process_image(image_batch):
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #print(image_batch)
    # Load the image
    img_list_transformed = []
    for index in image_batch:
        for img in index:
            #print(img)
            if img == '':
                image = Image.new('RGB', (256, 256), color='white')
            else:
                image = Image.open(img).convert("RGB")
            img_transform = image_transform(image)
            img_list_transformed.append(img_transform)
    stacked_images = torch.stack(img_list_transformed, dim=0)
    
    return stacked_images