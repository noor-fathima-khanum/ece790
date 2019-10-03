import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import seaborn as sns
sns.set()

import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

class BaseMLP(nn.Module):
    """
    A multi-layer perceptron model for MNIST. Consists of three fully-connected
    layers, the first two of which are followed by a ReLU.
    """

    def __init__(self):
        super().__init__()
        in_size = 784

        self.nn = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, in_data):
        return self.nn(in_data)

model = BaseMLP().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

checkpoint = torch.load('./mnist_saved_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()