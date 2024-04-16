#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info">
# <b>Deadline:</b> March 15, 2023 (Wednesday) 23:00
# </div>
# 
# # Exercise 2. Convolutional networks. VGG-style network.
# 
# In the second part you need to train a convolutional neural network with an architecture inspired by a VGG-network [(Simonyan \& Zisserman, 2015)](https://arxiv.org/abs/1409.1556).

# In[25]:


skip_training = True  # Set this flag to True before validation and submission


# In[26]:


# During evaluation, this cell sets skip_training to True
# skip_training = True

import tools, warnings
warnings.showwarning = tools.customwarn


# In[27]:


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


# In[28]:


# When running on your own computer, you can specify the data directory by:
# data_dir = tools.select_data_dir('/your/local/data/directory')
data_dir = tools.select_data_dir()


# In[29]:


# Select the device for training (use GPU if you have one)
#device = torch.device('cuda:0')
device = torch.device('cpu')


# In[30]:


if skip_training:
    # The models are always evaluated on CPU
    device = torch.device("cpu")


# ## FashionMNIST dataset
# 
# Let us use the FashionMNIST dataset. It consists of 60,000 training images of 10 classes: 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'.

# In[31]:


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


# # VGG-style network
# 
# Let us now define a convolution neural network with an architecture inspired by the [VGG-net](https://arxiv.org/abs/1409.1556).
# 
# The architecture:
# - A block of three convolutional layers with:
#     - 3x3 kernel
#     - 20 output channels
#     - one pixel zero-pading on both sides
#     - 2d batch normalization after each convolutional layer
#     - ReLU nonlinearity after each 2d batch normalization layer
# - Max pooling layer with 2x2 kernel and stride 2.
# - A block of three convolutional layers with:
#     - 3x3 kernel
#     - 40 output channels
#     - one pixel zero-pading on both sides
#     - 2d batch normalization after each convolutional layer
#     - ReLU nonlinearity after each 2d batch normalization layer
# - Max pooling layer with 2x2 kernel and stride 2.
# - One convolutional layer with:
#     - 3x3 kernel
#     - 60 output channels
#     - *no padding*
#     - 2d batch normalization after the convolutional layer
#     - ReLU nonlinearity after the 2d batch normalization layer
# - One convolutional layer with:
#     - 1x1 kernel
#     - 40 output channels
#     - *no padding*
#     - 2d batch normalization after the convolutional layer
#     - ReLU nonlinearity after the 2d batch normalization layer
# - One convolutional layer with:
#     - 1x1 kernel
#     - 20 output channels
#     - *no padding*
#     - 2d batch normalization after the convolutional layer
#     - ReLU nonlinearity after the 2d batch normalization layer
# - Global average pooling (compute the average value of each channel across all the input locations):
#     - 5x5 kernel (the input of the layer should be 5x5)
# - A fully-connected layer with 10 outputs (no nonlinearity)
# 
# Notes:
# * Batch normalization is expected to be right after a convolutional layer, before nonlinearity.
# * We recommend that you check the number of modules with trainable parameters in your network.

# In[32]:


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        # YOUR CODE HERE
        #Block 3conv, 20 output channels, 3x3 kernel, 1 pixel zero padding
        self.conv11 = nn.Conv2d(1,20,3, padding=1,padding_mode='zeros')
        self.bano11 = nn.BatchNorm2d(20)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(20,20,3, padding=1,padding_mode='zeros')
        self.bano12 = nn.BatchNorm2d(20)
        self.relu12 = nn.ReLU()
        self.conv13 = nn.Conv2d(20,20,3, padding=1,padding_mode='zeros')
        self.bano13 = nn.BatchNorm2d(20)
        self.relu13 = nn.ReLU()
        
        # Max pooling layer with 2x2 kernel and stride 2 
        self.mapo2 = nn.MaxPool2d(2,stride=2)
        
        #Block 3conv, 40 output channels, 3x3 kernel, 1 pixel zero padding
        self.conv31 = nn.Conv2d(20,40,3, padding=1,padding_mode='zeros')
        self.bano31 = nn.BatchNorm2d(40)
        self.relu31 = nn.ReLU()
        self.conv32 = nn.Conv2d(40,40,3, padding=1,padding_mode='zeros')
        self.bano32 = nn.BatchNorm2d(40)
        self.relu32 = nn.ReLU()
        self.conv33 = nn.Conv2d(40,40,3, padding=1,padding_mode='zeros')
        self.bano33 = nn.BatchNorm2d(40)
        self.relu33 = nn.ReLU()
        
        # Max pooling layer with 2x2 kernel and stride 
        self.mapo4 = nn.MaxPool2d(2,stride=2)
        
        # Convolutional layer , 60 output channels, 3x3 kernel,no padding 
        self.conv5 = nn.Conv2d(40,60,3)
        self.bano5 = nn.BatchNorm2d(60)
        self.relu5 = nn.ReLU()
        
        # Convolutional layer , 40 output channels, 1x1 kernel,no padding
        self.conv6 = nn.Conv2d(60,40,1)
        self.bano6 = nn.BatchNorm2d(40)
        self.relu6 = nn.ReLU()
        
        # Convolutional layer , 20 output channels, 1x1 kernel,no padding
        self.conv7 = nn.Conv2d(40,20,1)
        self.bano7 = nn.BatchNorm2d(20)
        self.relu7 = nn.ReLU()
        
        # Global average pooling with 5x5 kernel
        self.avpo8 = nn.AvgPool2d(5)
        
        #Output layers 
        self.line9 = nn.Linear(20, 10)
        
#         raise NotImplementedError()

    def forward(self, x, verbose=False):
        """
        Args:
          x of shape (batch_size, 1, 28, 28): Input images.
          verbose: True if you want to print the shapes of the intermediate variables.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        # YOUR CODE HERE
        h11 = self.relu11(self.bano11(self.conv11(x)))
        h12 = self.relu12(self.bano12(self.conv12(h11)))
        h13 = self.relu13(self.bano13(self.conv13(h12)))
        
        h2  = self.mapo2(h13)
        
        h31 = self.relu31(self.bano31(self.conv31(h2)))
        h32 = self.relu32(self.bano32(self.conv32(h31)))
        h33 = self.relu33(self.bano33(self.conv33(h32)))
        
        h4  = self.mapo2(h33)
        
        h5  = self.relu5(self.bano5(self.conv5(h4)))
        
        h6  = self.relu6(self.bano6(self.conv6(h5)))
        
        h7  = self.relu7(self.bano7(self.conv7(h6)))
        
        h8  = self.avpo8(h7)
             
        y = self.line9(torch.flatten(h8, 1))

        return(y)
        
        raise NotImplementedError()


# In[33]:


def test_VGGNet_shapes():
    net = VGGNet()
    net.to(device)

    # Feed a batch of images from the training data to test the network
    with torch.no_grad():
        images, labels = next(iter(trainloader))
        images = images.to(device)
        print('Shape of the input tensor:', images.shape)

        y = net(images, verbose=True)
        assert y.shape == torch.Size([trainloader.batch_size, 10]), f"Bad y.shape: {y.shape}"

    print('Success')

test_VGGNet_shapes()


# In[34]:


# Check the number of layers
def test_vgg_layers():
    net = VGGNet()
    
    # get gradients for parameters in forward path
    net.zero_grad()
    x = torch.randn(1, 1, 28, 28)
    outputs = net(x)
    outputs[0,0].backward()

    n_conv_layers = sum(1 for module in net.modules()
                        if isinstance(module, nn.Conv2d) and next(module.parameters()).grad is not None)
    assert n_conv_layers == 9, f"Wrong number of convolutional layers ({n_conv_layers})"

    n_bn_layers = sum(1 for module in net.modules()
                      if isinstance(module, nn.BatchNorm2d) and next(module.parameters()).grad is not None)
    assert n_bn_layers == 9, f"Wrong number of batch norm layers ({n_bn_layers})"

    n_linear_layers = sum(1 for module in net.modules()
                          if isinstance(module, nn.Linear) and next(module.parameters()).grad is not None)
    assert n_linear_layers == 1, f"Wrong number of linear layers ({n_linear_layers})"

    print('Success')

def test_vgg_net():
    net = VGGNet()
    
    # get gradients for parameters in forward path
    net.zero_grad()
    x = torch.randn(1, 1, 28, 28)
    outputs = net(x)
    outputs[0,0].backward()
    
    parameter_shapes = sorted(tuple(p.shape) for p in net.parameters() if p.grad is not None)
    print(parameter_shapes)
    expected = [
        (10,), (10, 20), (20,), (20,), (20,), (20,), (20,), (20,), (20,), (20,), (20,),
        (20,), (20,), (20,), (20, 1, 3, 3), (20, 20, 3, 3), (20, 20, 3, 3), (20, 40, 1, 1),
        (40,), (40,), (40,), (40,), (40,), (40,), (40,), (40,), (40,), (40,), (40,), (40,),
        (40, 20, 3, 3), (40, 40, 3, 3), (40, 40, 3, 3), (40, 60, 1, 1), (60,), (60,), (60,),
        (60, 40, 3, 3)]
    assert parameter_shapes == expected, "Wrong number of training parameters."
    
    print('Success')

test_vgg_layers()
test_vgg_net()


# # Train the network

# In[35]:


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
# Your task is to implement the training loop. The recommended hyperparameters:
# * Adam optimizer with learning rate 0.01.
# * Cross-entropy loss. Note that we did not use softmax nonlinearity in the final layer of our network. Therefore, we need to use a loss function with log_softmax implemented, such as [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss).
# * Number of epochs: 10
# 
# We recommend you to use function `compute_accuracy()` defined above to track the accaracy during training. The test accuracy should be above 0.89.
# 
# **Note: function `compute_accuracy()` sets the network into the evaluation mode which changes the way the batch statistics are computed in batch normalization. You need to set the network into the training mode (by calling `net.train()`) when you want to perform training.**

# In[20]:


net = VGGNet()


# In[21]:


# Implement the training loop in this cell
if not skip_training:
    # YOUR CODE HERE
    learning_rate = 0.01
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    n_epochs = 1 if skip_training else 10
    
    for i in range(n_epochs):
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


# In[24]:


# Save the model to disk (the pth-files will be submitted automatically together with your notebook)
# Set confirm=False if you do not want to be asked for confirmation before saving.
if not skip_training:
    tools.save_model(net, '2_vgg_net.pth', confirm=True)


# In[ ]:


if skip_training:
    net = VGGNet()
    tools.load_model(net, '2_vgg_net.pth', device)


# In[23]:


# Compute the accuracy on the test set
accuracy = compute_accuracy(net, testloader)
print(f'Accuracy of the VGG net on the test images: {accuracy: .3f}')
assert accuracy > 0.89, 'Poor accuracy'
print('Success')


# In[ ]:




