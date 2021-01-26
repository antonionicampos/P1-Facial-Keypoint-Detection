## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## Input is a image with 224x224 pixels
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Convolutional Layer 1: in_channels = 1, out_channels = 32, kernel_size = 5, stride = 1 (default)
        # Output Image: 220x220 after pooling 110x110 
        self.conv1 = nn.Conv2d(1, 32, 5)

        # Convolutional Layer 2: in_channels = 32, out_channels = 64, kernel_size = 4, stride = 1 (default)
        # Output Image: 106x106 after pooling 53x53
        self.conv2 = nn.Conv2d(32, 64, 5)

        # Convolutional Layer 3: in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1 (default)
        # Output Image: 51x51 after pooling 25x25
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # Convolutional Layer 4: in_channels = 128, out_channels = 256, kernel_size = 2, stride = 1 (default)
        # Output Image: 24x24 after pooling 12x12
        self.conv4 = nn.Conv2d(128, 256, 2)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers 
        # (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, stride=2)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)

        self.fc1 = nn.Linear(256*12*12, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.dropout1(self.pool(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool(F.relu(self.conv2(x))))
        x = self.dropout3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout4(self.pool(F.relu(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout5(F.relu(self.fc1(x)))
        x = self.dropout6(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x