import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    A basic building block for a VGG-like architecture.
    This block consists of a convolutional layer followed by batch normalization, 
    a ReLU activation, and optionally a MaxPooling layer. These blocks are stacked 
    to form deeper layers, progressively extracting features from the input.
    """
    
    def __init__(self, in_channels, out_channels, pool=False):
        """
        Initialize the ConvBlock.
        Arguments:
        - in_channels: Number of input channels (e.g., 3 for RGB images).
        - out_channels: Number of output channels (filters) for the convolutional layer.
        - pool: Whether to apply MaxPooling after the convolution (default is False).
        """

        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class VGG11(nn.Module):
    """
    A deep CNN based on the VGG11 architecture. It includes 8 convolutional layers 
    with ReLU activation and batch normalization, followed by max pooling. 
    The fully connected layers at the end perform classification with dropout 
    for regularization.
    """

    def __init__(self):
        super(VGG11, self).__init__()
        self.convblock_layers = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256, pool=True),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512, pool=True)
        )
        
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4 * 4 * 512, 2048),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(2048, 10))

    def forward(self, x):
        x = self.convblock_layers(x)
        x = x.view(-1, 4 * 4 * 512)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x