from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn.functional as F 
from torch.utils.data import TensorDataset, DataLoader



import matplotlib.pyplot as plt
import time
import os
import copy


def train(model, loader, optimizer, loss_fn, epochs=30, device=torch.device('cpu')):
    model.train()
    model = model.to(device)
    for epoch in range(int(epochs)):
        for i, (batch_data, batch_target) in enumerate(loader):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)

            optimizer.zero_grad()
            mse_loss = loss_fn(model(batch_data).float(), batch_target.view(-1, 1).float())
            mse_loss.backward()
            optimizer.step()
    return model

def evaluate(model, valid_loader, loss_fn, device=None):
    valid_loss = 0
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for valid_i, (data, targets) in enumerate(valid_loader):
            data, targets = data.to(device), targets.to(device)            
            pred = model(data)
            valid_loss += loss_fn(model(data), targets.view(-1, 1))
    return valid_loss.item()


# class CNN_Regressor(nn.Module):
class CNN_Regressor(nn.Module):
	def __init__(self, device=None):
		super(CNN_Regressor, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, 3)
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.conv3 = nn.Conv2d(64, 64, 3)
		self.bn1 = nn.BatchNorm2d(32)
		self.bn2 = nn.BatchNorm2d(64)
		# self.bn3 = nn.BatchNorm2d(64)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(64 * 4 * 4, 64)
		self.fc2 = nn.Linear(64, 10)
		self.fc3 = nn.Linear(10, 1)
	def forward(self, x):
		x = self.pool(self.bn1( F.relu(self.conv1(x))))
		x = self.pool(self.bn2(F.relu(self.conv2(x))))
		x = F.relu(self.conv3(x))
		x = x.view(-1, 64 * 4 * 4)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.fc3(x)

	def extract(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = F.relu(self.conv3(x))
		x = x.view(-1, 64 * 4 * 4)
		x = F.relu(self.fc1(x))
		return F.relu(self.fc2(x))


data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.6502, 0.4878, 0.4164], [0.1732, 0.1568, 0.1486])
    ])



# https://susanqq.github.io/UTKFace/
data_dir = 'data/Face_Age/'
dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

print(dir(dataset))

for i, sample in enumerate(dataset.samples):
	file_path, _ = sample
	age = file_path.split('/')[-1].split('_')[0]
	sample = tuple((file_path, float(age)))
	dataset.samples[i] = sample
	dataset.targets[i] = float(age)

indices = torch.arange(len(dataset))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(indices, indices, test_size=0.4, random_state=42)

print("Train Valid length:", len(X_train), len(X_test))

train_loader = DataLoader(dataset, batch_size=64, sampler=SubsetRandomSampler(X_train))
valid_loader = DataLoader(dataset, batch_size=128, sampler=SubsetRandomSampler(X_test))

device =  torch.device('cuda')
model = CNN_Regressor(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()
E = 50


valid_loss = evaluate(model, valid_loader, loss_fn, device)
print("Valid loss before training:", valid_loss)
model = train(model, train_loader, optimizer, loss_fn, E, device )
valid_loss = evaluate(model, valid_loader, loss_fn, device)
print("Valid loss after training:", valid_loss)



"""  BLR on features of trained NN Regressor """

extracted_Xs = []
Ys = []

model.eval()
model = model.to(device)
with torch.no_grad():
    for valid_i, (data, targets) in enumerate(valid_loader):
        data, targets = data.to(device), targets.to(device)
        features = model.extract(data)

        extracted_Xs.append(features)
        Ys.append(targets)

extracted_Xs = torch.cat(extracted_Xs)
Ys = torch.cat(Ys)
print(extracted_Xs.shape, Ys.shape)


import pandas as pd
pd.DataFrame(extracted_Xs.detach().cpu().numpy()).to_csv('data/Face_Age/face_age-CNN_features.csv', index=False)
pd.DataFrame(Ys.detach().cpu().numpy()).to_csv('data/Face_Age/face_age-labels.csv', index=False)

exit()





def compute_mean_std(dataset):

	loader = DataLoader(dataset, batch_size=128, num_workers=6)

	nimages = 0
	mean = 0.
	std = 0.
	for batch, _ in loader:
	    # Rearrange batch to be the shape of [B, C, W * H]
	    batch = batch.view(batch.size(0), batch.size(1), -1)
	    # Update total number of images
	    nimages += batch.size(0)
	    # Compute mean and std here
	    mean += batch.mean(2).sum(0) 
	    std += batch.std(2).sum(0)

	# Final step
	mean /= nimages
	std /= nimages

	print(mean)
	print(std)
	return mean, std