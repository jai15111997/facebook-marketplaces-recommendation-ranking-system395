from PIL import Image # Import necessary Packages
import torch
import torchvision.transforms as transforms

# Check if this script is being run directly
if __name__ == "__main__":
    print('api.py first!')

class image_utility:

    def __init__(self):

        # Initialize image_utility class
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) # Image Transformation to Tensor

    def process_image(self, image_batch):
        
        # Method to process a batch of images
        img_list_transformed = [] # Initialize list to store transformed images

        for index in image_batch: # Iterate through each image in the batch

            for img in index: # Iterate through each image path in the batch
                
                # Open image and convert it to RGB format, or create a white image if path is empty
                image = Image.new('RGB', (256, 256), color='white')
               
                if img != '':
                    image = Image.open(img).convert("RGB")
                
                # Apply image transformation pipeline
                img_transform = self.image_transform(image)

                # Append transformed image to the list
                img_list_transformed.append(img_transform) 

        # Stack transformed images along a new dimension (batch dimension)
        stacked_images = torch.stack(img_list_transformed, dim=0) 
        
        return stacked_images # Return batch of transformed images