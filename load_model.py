import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import fc_model
checkpoint = torch.load('checkpoint.pth')
#model = fc_model.Network(784, 10, [512, 256, 128])
model = fc_model.Network(checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'])
model.load_state_dict(checkpoint['state_dict'])

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}