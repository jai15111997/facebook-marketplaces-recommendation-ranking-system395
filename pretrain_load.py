import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Pretrained(torch.nn.Module):
    def __init__(self, dataset, dataloader):
        super().__init__()
        self.resnet_model = models.resnet50(pretrained=True)
        self.resnet_model.fc = nn.Linear(2048, 13)
        self.dataset = dataset
        self.dataloader = dataloader

    def forward(self, inp):
        return (self.resnet_model(inp))

        #works if you used torch.nn.Sequential for layers
    
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
        device = torch.device("cuda")
        #F = nn.functional.cross_entropy()
        #F = torch.nn.MSELoss()
        batch_idx = 0
        for epoch in range(epochs):
            for batch in dataloader:
                features, labels = batch
                predictions = self.resnet_model(features)
                #print(predictions.shape)
                #print(labels.shape)
                loss = nn.functional.cross_entropy(predictions, labels.long())
                print(loss.item())
                loss.backward()
                optmiser.step()
                optmiser.zero_grad()
                writer.add_scalar('loss', loss.item(), batch_idx)
                batch_idx += 1

            Pretrained.save_checkpoint(epoch)
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