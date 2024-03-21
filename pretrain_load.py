from datetime import datetime # Import datetime module for timestamping
from image_processor import image_utility # Import image utility class
import json 
import os # Import os module for file and directory operations
import torch 
from torchvision import models 
import torch.nn as nn # Import torch.nn module for neural network layers
import torch.optim as optim # Import torch.optim module for optimization algorithms
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter from torch.utils.tensorboard

# Instantiate image utility object
img_util = image_utility()

# Check if this script is being run directly
if __name__ == "__main__":
    print('Run main.py first!')

class Pretrained(torch.nn.Module):

    def __init__(self, dataset, dataloader):

        super().__init__()
        resnet_model_initial = models.resnet50(pretrained=True) # Load pre-trained ResNet-50 model
        
        # Freeze all layers except the last two
        for name, param in resnet_model_initial.named_parameters():

            if 'layers.4.2' in name or 'layers.4.1' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        added_layers = nn.Sequential(nn.ReLU(), nn.Linear(1000, 14)) # Add custom layers for classification
        self.resnet_model = nn.Sequential(resnet_model_initial, added_layers) # Combine pre-trained and custom layers
        self.dataset = dataset
        self.dataloader = dataloader
        self.img_embedding_dict = {} # Initialize dictionary for image embeddings

    def forward(self, inp):

        # Forward Pass through the Model
        features = self.resnet_model(inp)
        return features
    
    def save_checkpoint(self, epoch, folder_path='model_evaluation'):

        """
        Save the model checkpoint.
        epoch: Current epoch number
        folder_path: Path to the folder where model checkpoints will be saved
        """

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Get current timestamp
        model_folder = os.path.join(folder_path, f'model_{timestamp}') # Create model folder path
        os.makedirs(model_folder, exist_ok=True) # Create model folder if it doesn't exist

        weights_folder = os.path.join(model_folder, 'weights') # Create weights folder path
        os.makedirs(weights_folder, exist_ok=True) # Create weights folder if it doesn't exist

        checkpoint_path = os.path.join(weights_folder, f'epoch_{epoch}_weights.pth') # Create checkpoint path
        torch.save(self.resnet_model.state_dict(), checkpoint_path) # Save model state dict to checkpoint path

    def train(self, dataloader, validation_dl, epochs):

        """
        Train the model.
        dataloader: DataLoader for training data
        validation_dl: DataLoader for validation data
        epochs: Number of epochs to train for
        """

        optmiser = optim.Adam(self.resnet_model.parameters(), lr = 0.001)
        writer = SummaryWriter() # Create SummaryWriter for TensorBoard logging
        device = torch.device("cpu") # Set device to CPU
        global_step = 0 # Initialize global step counter

        for epoch in range(epochs):

            for batch in dataloader:

                batch_images, labels = batch
                features = img_util.process_image(batch_images) # Process images
                predictions = self.resnet_model(features) # Get model predictions
                self.img_embedding_dict.update(img_util.dict_updater(batch_images, predictions)) # Update embedding dict
                loss = nn.functional.cross_entropy(predictions, labels) # Compute loss
                print(loss.item())
                loss.backward() # Backpropagation
                optmiser.step() # Update weights
                optmiser.zero_grad() # Reset gradients
                writer.add_scalar('loss', loss.item(), global_step) # Log loss to TensorBoard
                global_step += 1 # Increment global step

            self.save_checkpoint(epoch) # Save model checkpoint
            self.resnet_model.eval() # Set model to evaluation mode
            correct = 0 # Set model to evaluation mode
            total = 0 # Initialize total predictions counter

            with torch.no_grad(): # Disable gradient calculation for validation

                for input_names, labels_names in validation_dl:

                    input_tensors = img_util.process_image(input_names) # Process input images
                    labels_tensors = torch.tensor(labels_names) # Convert labels to tensor
                    inputs, labels = input_tensors.to(device), torch.tensor(labels_tensors).to(device) # Move data to device
                    outputs = self.resnet_model(inputs) # Get model outputs
                    _, predicted = torch.max(outputs.data, 1) # Get predicted labels
                    total += labels.size(0) # Increment total count
                    correct += (predicted == labels).sum().item() # Increment correct count
                    accuracy = correct / total # Calculate accuracy
                    print(f'accuracy = {accuracy}')
        
        with open('image_embeddings.json', 'w') as json_file:
            
            # Save image embeddings to JSON file
            json.dump(self.img_embedding_dict, json_file) 