import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import seaborn as sns
from torch.utils.data.sampler import SubsetRandomSampler
sns.set()

import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

#########################################################################################################

# Database Download
batch_size = 64

train_dataset = datasets.MNIST('./data', 
                               train=True, 
                               download=True,
                               transform=transforms.ToTensor())

validation_dataset = datasets.MNIST('./data', 
                               train=True, 
                               download=True, 
                               transform=transforms.ToTensor())
                               
test_dataset = datasets.MNIST('./data', 
                                    train=False, 
                                    transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False)

valid_size=0.1
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
len_train=num_train-split
print ("NUM TRAIN,SPLIT",num_train,split)
np.random.seed(35)
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=batch_size, sampler=train_sampler)

validation_loader = torch.utils.data.DataLoader(validation_dataset, 
                    batch_size=batch_size, sampler=valid_sampler)
                    
for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

####################################################################################################

# MLP Model definition
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


def encoder(in_data):
   return torch.sum(in_data, dim=1).view(in_data.size(0),1,-1)

def decoder(in_data,num_in):
    out = in_data[:, -1] - torch.sum(in_data[:, :-1], dim=1)
    out = out.unsqueeze(1).repeat(1, num_in, 1)
    print ("OUTSIZE",out.size())
    return out

def add_labels(in_data):
    return torch.sum(in_data, dim=1)

def decode_possibilities(parity_output,f_output,base_model_output_dim,k,r):
   new_parity_output = parity_output.view(parity_output.size(0), -1, parity_output.size(-1))
   in_decode = torch.cat((f_output, new_parity_output), dim=1)
   
   # Erasure Mask Creation
   erased_indices = [torch.LongTensor([e]) for e in range(k)]
   erase_mask = torch.ones((len(erased_indices),k+r,base_model_output_dim))
   acc_mask = torch.zeros((len(erased_indices),k)).byte()
   for i, erased_idx in enumerate(erased_indices):
       i = torch.LongTensor([i])
       erase_mask[i, erased_idx, :] = 0.
       acc_mask[i, erased_idx] = 1
   
   _, num_in, dim = in_decode.size()
   mb_emask = erase_mask.repeat(parity_output.size(0), 1, 1)
   mb_amask = acc_mask.repeat(batch_size, 1)
   
   num_erasure_scenarios = erase_mask.size(0)
   in_decode = in_decode.repeat(1, num_erasure_scenarios, 1).view(parity_output.size(0) * num_erasure_scenarios, num_in, dim)
   in_decode = in_decode * mb_emask
   print ("in_decode",in_decode)
   print ("new_parity_output,in_decode",new_parity_output.size(),in_decode.size())
   return in_decode,num_in
   
################################################################################################

# Load the MNIST base-model weights
f_model = BaseMLP().to(device)
checkpoint = torch.load('./mnist_saved_model.pth', map_location=torch.device('cpu'))
f_model.load_state_dict(checkpoint['model_state_dict'])
f_model.eval()

# Define Parity Model and the training hyperparameters
parity_model = BaseMLP().to(device)
optimizer = torch.optim.SGD(parity_model.parameters(), lr=0.01, momentum=0.5)
#criterion = nn.CrossEntropyLoss()
loss_criterion=torch.nn.MSELoss()

k=2
r=1
base_model_output_dim=10
## Training Model
def train(epoch, log_interval=200):
    # Set model to training mode
    parity_model.train()  
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.view(-1, 28*28)
        print ("data after view",data.size())
        data = data.to(device)
        target = target.to(device)
        
        # Pass data through the network
        f_output = f_model(data)
        
        ## Change data view for parity
        data = data.view(-1,k,28*28)
        f_output = f_output.view(-1,k,f_output.size(1))
        target = target.view(-1,k)
        print ("VIEW FOR PARITY",data.size(),f_output.size(),target.size())
        
        parity_encoded = encoder(data)
        print ("PARITY ENCODED",parity_encoded.size())
        parity_encoded_output = parity_model(parity_encoded)
        print ("parity_encoded_output",parity_encoded_output.size())
        parity_encoded_output = parity_encoded_output.view(parity_encoded_output.size(0),parity_encoded_output.size(-1))
        print ("parity_encoded_output2",parity_encoded_output.size())
        
        # Sum up Target Labels for parity to compute loss
        target_parity_encoded_output=add_labels(f_output)
        print ("target_parity_encoded_output",target_parity_encoded_output.size())
        
        # Zero gradient buffers
        optimizer.zero_grad()
        
        loss = loss_criterion(parity_encoded_output,target_parity_encoded_output)
        print ("LOSS",loss)
        
        ## Decode different possibilities
        decoded_parity_possibilities,num_in=decode_possibilities(parity_encoded_output,f_output,base_model_output_dim,k,r)
        parity_decoded_output=decoder(decoded_parity_possibilities,k)
        
        # Calculate Accuracy
        _, num_out, out_dim = f_output.size()
        
        exit()
        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        #if batch_idx % log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len_train,
        #        100. * batch_idx / len(train_loader), loss.data.item()))
    
    #torch.save({
    #        'epoch': epoch,
    #        'model_state_dict': model.state_dict(),
    #        'optimizer_state_dict': optimizer.state_dict(),
    #        'loss': loss,
    #        }, './mnist_saved_model.pth')

## Validating Model
def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.view(-1, 28*28)
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)
    
    print ("CORRECT",correct)
    accuracy = 100. * correct.to(torch.float32) / split
    accuracy_vector.append(accuracy)
    
    print('\nvalidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, split, accuracy))

epochs = 50

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)


