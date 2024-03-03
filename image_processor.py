import torch
from PIL import Image
import torchvision.transforms as transforms

if __name__ == "__main__":
    print('Run main.py first!')

class image_utility:

    def __init__(self):
        self.dict = {}
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process_image(self, image_batch):
        
        #print(image_batch)
        # Load the image
        img_list_transformed = []
        for index in image_batch:
            for img in index:
                #print(img)
                image = Image.new('RGB', (256, 256), color='white')
                if img != '':
                    image = Image.open(img).convert("RGB")
                img_transform = self.image_transform(image)
                img_list_transformed.append(img_transform)
        stacked_images = torch.stack(img_list_transformed, dim=0)
        
        return stacked_images

    def dict_updater(self, image_batch, predictions):
        i = 0
        for index in image_batch:
            for img in index:
                image = Image.new('RGB', (256, 256), color='white')
                if img != '':
                    image = Image.open(img).convert("RGB")
                self.dict[img] = predictions[i]
                i += 1
        return self.dict