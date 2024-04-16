#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info">
# <b>Deadline:</b> March 15, 2023 (Wednesday) 23:00
# </div>
# 
# # Exercise 3. Convolutional networks. ResNet.
# 
# In the third part you need to train a convolutional neural network with a ResNet architecture [(He et al, 2016)](https://arxiv.org/abs/1512.03385).

# In[1]:


skip_training = True  # Set this flag to True before validation and submission


# In[2]:


# During evaluation, this cell sets skip_training to True
# skip_training = True

import tools, warnings
warnings.showwarning = tools.customwarn


# In[3]:


import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tools
import tests


# In[4]:


# When running on your own computer, you can specify the data directory by:
# data_dir = tools.select_data_dir('/your/local/data/directory')
data_dir = tools.select_data_dir()


# In[5]:


# Select the device for training (use GPU if you have one)
#device = torch.device('cuda:0')
device = torch.device('cpu')


# In[6]:


if skip_training:
    # The models are always evaluated on CPU
    device = torch.device("cpu")


# ## FashionMNIST dataset
# 
# Let us use the FashionMNIST dataset. It consists of 60,000 training images of 10 classes: 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'.

# In[7]:


transform = transforms.Compose([
    transforms.ToTensor(),  # Transform to tensor
    transforms.Normalize((0.5,), (0.5,))  # Scale images to [-1, 1]
])

trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)


# ## ResNet
# 
# We create a network with an architecure inspired by [ResNet](https://arxiv.org/pdf/1512.03385.pdf).
# 
# ### ResNet block
# Our ResNet consists of blocks with two convolutional layers and a skip connection.
# 
# In the most general case, our implementation should have:
# 
# <img src="resnet_block_04.png" width=220 style="float: right;">
# 
# * Two convolutional layers with:
#     * 3x3 kernel
#     * no bias terms
#     * padding with one pixel on both sides
#     * 2d batch normalization after each convolutional layer.
# 
# * **The first convolutional layer also (optionally) has:**
#     * different number of input channels and output channels
#     * change of the resolution with stride.
# 
# * The skip connection:
#     * simply copies the input if the resolution and the number of channels do not change.
#     * if either the resolution or the number of channels change, the skip connection should have one convolutional layer with:
#         * 1x1 convolution **without bias**
#         * change of the resolution with stride (optional)
#         * different number of input channels and output channels (optional)
#     * if either the resolution or the number of channels change, the 1x1 convolutional layer is followed by 2d batch normalization.
# 
# * The ReLU nonlinearity is applied after the first convolutional layer and at the end of the block.
# 
# <div class="alert alert-block alert-warning">
# <b>Note:</b> Batch normalization is expected to be right after a convolutional layer.
# </div>

# <img src="resnet_blocks_123.png" width=650 style="float: top;">
# 
# The implementation should also handle specific cases such as:
# 
# Left: The number of channels and the resolution do not change.
# There are no computations in the skip connection.
# 
# Middle: The number of channels changes, the resolution does not change.
# 
# Right: The number of channels does not change, the resolution changes.

# Your task is to implement this block. You should use the implementations of layers in `nn.Conv2d`, `nn.BatchNorm2d` as the tests rely on those implementations.

# In[23]:


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(Block, self).__init__()
        # YOUR CODE HERE
        #Two Conv Layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, padding_mode='zeros', bias=False)
        self.bano1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels,out_channels, 3, padding=1,padding_mode='zeros', bias=False)
        self.bano2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.bano3 = nn.BatchNorm2d(out_channels)
        
        self.relu4 = nn.ReLU()
        
        self.skipconnection = 0 if stride==1 and in_channels==out_channels else 1
#         raise NotImplementedError()

    def forward(self, x):
        # YOUR CODE HERE
        h11 = self.relu1(self.bano1(self.conv1(x)))
        h12 = self.bano2(self.conv2(h11))
        
        h2  = self.bano3(self.conv3(x))
        h3 = x if self.skipconnection == 0 else h2
        
        y = self.relu4(h12+h3)
        
        return(y)
        raise NotImplementedError()


# In[24]:


def test_Block_shapes():

    # The number of channels and resolution do not change
    batch_size = 20
    x = torch.zeros(batch_size, 16, 28, 28)
    block = Block(in_channels=16, out_channels=16)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 16, 28, 28]), "Bad shape of y: y.shape={}".format(y.shape)

    # Increase the number of channels
    block = Block(in_channels=16, out_channels=32)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 32, 28, 28]), "Bad shape of y: y.shape={}".format(y.shape)

    # Decrease the resolution
    block = Block(in_channels=16, out_channels=16, stride=2)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 16, 14, 14]), "Bad shape of y: y.shape={}".format(y.shape)

    # Increase the number of channels and decrease the resolution
    block = Block(in_channels=16, out_channels=32, stride=2)
    y = block(x)
    assert y.shape == torch.Size([batch_size, 32, 14, 14]), "Bad shape of y: y.shape={}".format(y.shape)

    print('Success')

test_Block_shapes()


# In[25]:


tests.test_Block(Block)
tests.test_Block_relu(Block)
tests.test_Block_batch_norm(Block)


# ### Group of blocks
# 
# ResNet consists of several groups of blocks. The first block in a group may change the number of channels (often multiples the number by 2) and subsample (using strides).
# 
# <img src="resnet_group.png" width=500 style="float: left;">

# In[26]:


# We implement a group of blocks in this cell
class GroupOfBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, stride=1):
        super(GroupOfBlocks, self).__init__()

        first_block = Block(in_channels, out_channels, stride)
        other_blocks = [Block(out_channels, out_channels) for _ in range(1, n_blocks)]
        self.group = nn.Sequential(first_block, *other_blocks)

    def forward(self, x):
        return self.group(x)


# In[27]:


# Let's print a block
group = GroupOfBlocks(in_channels=10, out_channels=20, n_blocks=3)
print(group)


# ### ResNet
# 
# Next we implement a ResNet with the following architecture. It contains three groups of blocks, each group having two basic blocks.
# 
# <img src="resnet.png" width=900 style="float: left;">

# The cell below contains the implementation of our ResNet.

# In[28]:


class ResNet(nn.Module):
    def __init__(self, n_blocks, n_channels=64, num_classes=10):
        """
        Args:
          n_blocks (list):   A list with three elements which contains the number of blocks in 
                             each of the three groups of blocks in ResNet.
                             For instance, n_blocks = [2, 4, 6] means that the first group has two blocks,
                             the second group has four blocks and the third one has six blocks.
          n_channels (int):  Number of channels in the first group of blocks.
          num_classes (int): Number of classes.
        """
        assert len(n_blocks) == 3, "The number of groups should be three."
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.group1 = GroupOfBlocks(n_channels, n_channels, n_blocks[0])
        self.group2 = GroupOfBlocks(n_channels, 2*n_channels, n_blocks[1], stride=2)
        self.group3 = GroupOfBlocks(2*n_channels, 4*n_channels, n_blocks[2], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = nn.Linear(4*n_channels, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, verbose=False):
        """
        Args:
          x of shape (batch_size, 1, 28, 28): Input images.
          verbose: True if you want to print the shapes of the intermediate variables.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        if verbose: print(x.shape)
        x = self.conv1(x)
        if verbose: print('conv1:  ', x.shape)
        x = self.bn1(x)
        if verbose: print('bn1:    ', x.shape)
        x = self.relu(x)
        if verbose: print('relu:   ', x.shape)
        x = self.maxpool(x)
        if verbose: print('maxpool:', x.shape)

        x = self.group1(x)
        if verbose: print('group1: ', x.shape)
        x = self.group2(x)
        if verbose: print('group2: ', x.shape)
        x = self.group3(x)
        if verbose: print('group3: ', x.shape)

        x = self.avgpool(x)
        if verbose: print('avgpool:', x.shape)

        x = x.view(-1, self.fc.in_features)
        if verbose: print('x.view: ', x.shape)
        x = self.fc(x)
        if verbose: print('out:    ', x.shape)

        return x


# In[29]:


def test_ResNet_shapes():
    # Create a network with 2 block in each of the three groups
    n_blocks = [2, 2, 2]  # number of blocks in the three groups
    net = ResNet(n_blocks, n_channels=10)
    net.to(device)

    # Feed a batch of images from the training data to test the network
    with torch.no_grad():
        images, labels = next(iter(trainloader))
        images = images.to(device)
        print('Shape of the input tensor:', images.shape)

        y = net.forward(images, verbose=True)
        print(y.shape)
        assert y.shape == torch.Size([trainloader.batch_size, 10]), "Bad shape of y: y.shape={}".format(y.shape)

    print('Success')

test_ResNet_shapes()


# # Train the network

# In[30]:


# This function computes the accuracy on the test dataset
def compute_accuracy(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# ### Training loop
# 
# In the cell below, implement the training loop. The recommended hyperparameters:
# * Adam optimizer with learning rate 0.01.
# * Cross-entropy loss. Note that we did not use softmax nonlinearity in the final layer of our network. Therefore, we need to use a loss function with log_softmax implemented, such as [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss).
# * Number of epochs: 10
# 
# We recommend you to use function `compute_accuracy()` defined above to track the accaracy during training. The test accuracy should be above 0.9.
# 
# **Note: function `compute_accuracy()` sets the network into the evaluation mode which changes the way the batch statistics are computed in batch normalization. You need to set the network into the training mode (by calling `net.train()`) when you want to perform training.**

# In[31]:


# Create the network
n_blocks = [2, 2, 2]  # number of blocks in the three groups
net = ResNet(n_blocks, n_channels=16)
net.to(device)


# In[32]:


if not skip_training:
    # YOUR CODE HERE
    learning_rate = 0.01
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    n_epochs = 1 if skip_training else 10
    
    for i in range(n_epochs):
        net.train()
        for images, labels in trainloader:
            # Clear gradients
            optimizer.zero_grad()

            #Forward computations - Calculate Output and Loss
            output = net.forward(images)
            loss = criterion(output, labels)

            # Backward computations
            loss.backward()

            # Update parameter using optimizer
            optimizer.step()    # Does the update
            
        accuracy = compute_accuracy(net, testloader)
        print('Iteration_ %d' % i, '_Accuracy: %.3f' % accuracy)
    print('Trainning Finish')
    
#     raise NotImplementedError()


# In[34]:


# Save the model to disk (the pth-files will be submitted automatically together with your notebook)
# Set confirm=False if you do not want to be asked for confirmation before saving.
if not skip_training:
    tools.save_model(net, '3_resnet.pth', confirm=True)


# In[ ]:


if skip_training:
    net = ResNet(n_blocks, n_channels=16)
    tools.load_model(net, '3_resnet.pth', device)


# In[33]:


# Compute the accuracy on the test set
accuracy = compute_accuracy(net, testloader)
print('Accuracy of the network on the test images: %.3f' % accuracy)
n_blocks = sum(type(m) == Block for _, m in net.named_modules())
assert n_blocks == 6, f"Wrong number ({n_blocks}) of blocks used in the network."

assert accuracy > 0.9, "Poor accuracy ({:.3f})".format(accuracy)
print('Success')


# In[ ]:




