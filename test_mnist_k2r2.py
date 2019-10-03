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

checkpoint = torch.load('./mnist_saved_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


test_dataset = datasets.MNIST('./data', 
                                    train=False, 
                                    transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=2, 
                                                shuffle=False)

### Define Encoder and Decoder
def encoder(in_data):
   return torch.sum(in_data, dim=0).view(1,in_data.size(1))
   

def decoder(summed,x1):
   return (summed-x1)

predicted_vector=[]
parity_vector = []

model.eval()
test_loss, correct_pred = 0, 0
correct_parity=0
for data, target in test_loader:
    data = data.view(-1, 28*28)
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    pred = output.data.max(1)[1] # get the index of the max log-probability   
    predicted_vector.append(pred.data)
    
    #Encoder      
    encoded_data = encoder(data)
    encoded_output = model(encoded_data)
    
    #Decoder
    # Recover X1 which means X2 is available
    decodedx1_data = decoder(encoded_output,output[1,:])
    decodedx1_pred = decodedx1_data.data.max(1)[1]
    
    # Recover X2 which means X1 is available
    decodedx2_data = decoder(encoded_output,output[0,:])
    decodedx2_pred = decodedx2_data.data.max(1)[1]

    decoded_pred = torch.cat((decodedx1_pred,decodedx2_pred),0)
    parity_vector.append(decoded_pred.data)
    
    
      
    correct_pred += pred.eq(target.data).cpu().sum()
    correct_parity += decoded_pred.eq(target.data).cpu().sum()
    
    #val_loss /= len(validation_loader)
    #loss_vector.append(val_loss)


accuracy_pred = 100. * correct_pred.to(torch.float32) / len(test_loader.dataset)
accuracy_parity = 100. * correct_parity.to(torch.float32) / len(test_loader.dataset)

#accuracy_vector.append(accuracy)
    
print('\nTest set Normal Prediction: Accuracy: {}/{} ({:.0f}%)\n'.format(correct_pred, len(test_loader.dataset), accuracy_pred))
print('\nTest set Parity Prediction: Accuracy: {}/{} ({:.0f}%)\n'.format(correct_parity, len(test_loader.dataset), accuracy_parity))