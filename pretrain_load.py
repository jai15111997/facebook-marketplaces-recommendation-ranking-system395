import torch
from torchvision import  models
import torch.nn as nn
import torch.optim as optim
import os
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from image_processor import image_utility

img_util = image_utility()

if __name__ == "__main__":
    print('Run main.py first!')

class Pretrained(torch.nn.Module):
    def __init__(self, dataset, dataloader):
        super().__init__()
        resnet_model_initial = models.resnet50(pretrained=True)
        
        # Freeze all layers except the last two
        for name, param in resnet_model_initial.named_parameters():
            if 'layers.4.2' in name or 'layers.4.1' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        added_layers = nn.Sequential(nn.ReLU(), nn.Linear(1000, 14))
        self.resnet_model = nn.Sequential(resnet_model_initial, added_layers)
        #self.fc = nn.Linear(1, 1000)
        self.dataset = dataset
        self.dataloader = dataloader
        self.img_embedding_dict = {}

    def forward(self, inp):
        # Pass the input through the model
        features = self.resnet_model(inp)
        return features
    
    def save_checkpoint(self, epoch, folder_path='model_evaluation'):

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_folder = os.path.join(folder_path, f'model_{timestamp}')
        os.makedirs(model_folder, exist_ok=True)

        weights_folder = os.path.join(model_folder, 'weights')
        os.makedirs(weights_folder, exist_ok=True)

        checkpoint_path = os.path.join(weights_folder, f'epoch_{epoch}_weights.pth')
        torch.save(self.resnet_model.state_dict(), checkpoint_path)

    def train(self, dataloader, validation_dl, epochs):
        optmiser = optim.Adam(self.resnet_model.parameters(), lr = 0.001)
        writer = SummaryWriter()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_step = 0
        for epoch in range(epochs):
            for batch in dataloader:
                batch_images, labels = batch
                features = img_util.process_image(batch_images)
                predictions = self.resnet_model(features)
                self.img_embedding_dict.update(img_util.dict_updater(batch_images, predictions))
                #print(self.img_embedding_dict)
                #print(features.shape)
                #print(predictions.shape)
                #print(labels.shape)
                loss = nn.functional.cross_entropy(predictions, labels)
                print(loss.item())
                loss.backward()
                optmiser.step()
                optmiser.zero_grad()
                writer.add_scalar('loss', loss.item(), global_step)
                global_step += 1

            self.save_checkpoint(epoch)
            self.resnet_model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in validation_dl:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.resnet_model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    accuracy = correct / total
                    print(f'accuracy = {accuracy}')
        
        with open('image_embeddings.json', 'w') as json_file:
            json.dump(self.img_embedding_dict, json_file)