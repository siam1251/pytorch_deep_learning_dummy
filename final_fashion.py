import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import fc_model
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


# Create the network, define the criterion and optimizer

model = fc_model.Network(784, 10, [512, 256, 128])
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

fc_model.train(model, trainloader, testloader, criterion, optimizer, epochs=2)

## saving
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())
#torch.save(model.state_dict(), 'checkpoint.pth')
## save with input output and more
checkpoint={'input_size':784, 'output_size':10,
            'hidden_layers': [each.out_features for each in model.hidden_layers],
            'state_dict': model.state_dict() # 'state_dict' most important
            }
torch.save(checkpoint, 'checkpoint.pth')
# state_dict = torch.load('checkpoint.pth')
# print(state_dict.keys())
# model.load_state_dict(state_dict)