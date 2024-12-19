import torch.nn as nn
import torch.nn.functional as F

class ConvNet_1(nn.Module):
    """
    A simple Convolutional Neural Network with two convolutional layers,
    followed by three fully connected layers for classification. This model 
    uses ReLU activations and max pooling after each convolution.
    """

    def __init__(self):
        super(ConvNet_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ConvNet_2(nn.Module):
    """
    A deeper Convolutional Neural Network with batch normalization and dropout.
    This model uses 3 convolutional layers with batch normalization followed by 
    two fully connected layers and dropout for regularization.
    """

    def __init__(self):
        super(ConvNet_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
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

        