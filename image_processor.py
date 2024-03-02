import torch
from PIL import Image
import torchvision.transforms as transforms

def process_image(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Define the transformations (adjust according to your training transformations)
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformations
    processed_image = image_transform(image)

    # Add a batch dimension to make it a batch of size 1
    processed_image = processed_image.unsqueeze(0)

    return processed_image

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/your/image.jpg"
    processed_image = process_image(image_path)

    # Assuming p_train is your feature extraction model
    # You need to load the model before using it
    # p_train = Pretrained(...)  # Load your model here

    # Set the model to evaluation mode
    p_train.eval()

    # Pass the processed image through the model
    with torch.no_grad():
        features = p_train(processed_image)

    # Now 'features' contains the high-level abstract features extracted from the image
    print("Processed image shape:", processed_image.shape)
    print("Features shape:", features.shape)