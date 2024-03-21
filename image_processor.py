from PIL import Image # Import Essential Packages
import torch
import torchvision.transforms as transforms

# Check if this script is being run directly
if __name__ == "__main__":
    print('Run main.py first!')

class image_utility:

    def __init__(self):

        self.dict = {} # Initialize an empty dictionary
        # Define image transformations
        self.image_transform = transforms.Compose([
            transforms.Resize(256), # Resize image to 256x256
            transforms.CenterCrop(256), # Center crop image to 256x256
            transforms.ToTensor(), # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process_image(self, image_batch):
        
        """
        Process a batch of images.
        image_batch: List of image paths
        """

        img_list_transformed = [] # Initialize an empty list to store transformed images

        # Loop through each image in the batch
        for index in image_batch:

            for img in index:

                # Create a white image if path is empty, otherwise open the image
                image = Image.new('RGB', (256, 256), color='white')
                if img != '':
                    image = Image.open(img).convert("RGB")
                img_transform = self.image_transform(image) # Apply transformations to the image
                img_list_transformed.append(img_transform) # Append transformed image to the list

        stacked_images = torch.stack(img_list_transformed, dim=0) # Stack transformed images along batch dimension
        
        return stacked_images

    def dict_updater(self, image_batch, predictions):

        """
        Update dictionary with image predictions.
        image_batch: List of image paths
        predictions: List of predictions
        """

        i = 0 # Initialize counter

        # Loop through each image in the batch
        for index in image_batch:

            for img in index:

                if img == '':
                    img_name = 'NULL' # Set image name to 'NULL' if path is empty
                    self.dict[img_name] = predictions[i].tolist() # Update dictionary with prediction
                else:
                    img_name = img.split('/')[1] # Extract image name from path
                    self.dict[img_name] = predictions[i].tolist() # Update dictionary with prediction
                i += 1 # Increment counter
        return self.dict # Return the updated dictionary