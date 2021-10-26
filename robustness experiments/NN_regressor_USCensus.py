import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
data_fir = 'data/US_Census/acs2017_census_tract_data.csv' 
df = pd.read_csv(data_fir)

print(df.columns)
# df = df.drop(columns=['CensusTract', 'State', 'County'])
df = df.drop(columns=['TractId', 'State', 'County']).dropna()

print(df.describe())
print(df.info())
y = df['Income']
X = df.drop(columns=['Income'])

from sklearn.preprocessing import StandardScaler, minmax_scale

X = StandardScaler().fit_transform(X=X)
y = minmax_scale(y)

print("Data and labels shape:", X.shape, y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

""" Torch training """

import torch
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import TensorDataset, DataLoader


tensor_x = torch.Tensor(X_train) # transform to torch tensor
tensor_y = torch.Tensor(y_train)

dataset = TensorDataset(tensor_x,tensor_y) # create your datset
loader = DataLoader(dataset, batch_size=64) # create your dataloader


tensor_x = torch.Tensor(X_test) # transform to torch tensor
tensor_y = torch.Tensor(y_test)

dataset = TensorDataset(tensor_x, tensor_y) # create your datset
valid_loader = DataLoader(dataset, batch_size=1000) # create your dataloader

class NN_Regressor(nn.Module):
    def __init__(self, input_dim=8, output_dim=1, device=None):
        super(NN_Regressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def extract(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def train(model, loader, optimizer, loss_fn, epochs=30, device=torch.device('cpu')):

    model.train()
    model = model.to(device)
    for epoch in range(int(epochs)):
        for i, (batch_data, batch_target) in enumerate(loader):
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)

            optimizer.zero_grad()
            mse_loss = loss_fn(model(batch_data), batch_target.view(-1, 1))
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


device =  torch.device('cuda')
model = NN_Regressor(input_dim=X.shape[1], device=device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = torch.nn.MSELoss()
E = 50

valid_loss = evaluate(model, valid_loader, loss_fn, device)
print("Valid loss before training:", valid_loss)
model = train(model, loader, optimizer, loss_fn, E, device )
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

pd.DataFrame(extracted_Xs.detach().cpu().numpy()).to_csv('data/US_Census/USCensus-2017-NN_features.csv', index=False)
pd.DataFrame(Ys.detach().cpu().numpy()).to_csv('data/US_Census/USCensus-2017-labels.csv', index=False)

exit()
