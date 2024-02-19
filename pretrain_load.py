import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
from torchvision import datasets, models, transforms

class Pretrained(torch.nn.Module):
    def __init__(self, dataset, dataloader):
        super().__init__()
        self.resnet_model = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet_model.fc = torch.nn.Linear(2048, 13)
        self.dataset = dataset
        self.dataloader = dataloader

    def forward(self, inp):
        return self.resnet_model(inp)

        #works if you used torch.nn.Sequential for layers
    
    def train(self, dataloader, epochs):
        optmiser = torch.optim.Adam(self.resnet_model.parameters(), lr = 0.01)
        F = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for batch in dataloader:
                features, labels = batch
                predictions = self.resnet_model(features)
                #print(predictions.shape)
                #print(labels.shape)
                loss = F(predictions, labels)
                print(loss)
                loss.backward()
                optmiser.step()
                optmiser.zero_grad()
