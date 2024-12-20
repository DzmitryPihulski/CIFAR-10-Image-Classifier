import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    A basic building block for a ResNet-like architecture. 
    This block consists of two convolutional layers, each followed by batch 
    normalization and ReLU activations. It also includes a skip connection
    that adds the input to the output (residual connection).
    """

    def __init__(self, in_channels, out_channels, downsample=False):
        """
        Initialize the ConvBlock.
        Arguments:
        - in_channels: Number of input channels
        - out_channels: Number of output channels
        - downsample: Boolean indicating whether to downsample the spatial dimensions
        """
        
        stride = 2 if downsample else 1
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        
        self.expansion = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.expansion = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.batchnorm1(self.conv1(x)))
        out = self.batchnorm2(self.conv2(out))
        out = F.relu(out + self.expansion(x))
        return out
    
class ResNet18(nn.Module):
    """
    A ResNet-18 architecture with residual connections. 
    It uses ConvBlock layers to build the network. Each ConvBlock includes
    convolutional layers, batch normalization, and skip connections.
    """

    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.convblock_layers = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 128, downsample=True),  # Downsample spatial dimensions
            ConvBlock(128, 128),
            ConvBlock(128, 256, downsample=True),  # Further downsampling
            ConvBlock(256, 256),
            ConvBlock(256, 512, downsample=True),  # Further downsampling
            ConvBlock(512, 512)
        )
        self.linear = nn.Linear(512, 10)
    
    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.convblock_layers(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x