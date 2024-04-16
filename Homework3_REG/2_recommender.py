#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info">
# <b>Deadline:</b> March 22, 2023 (Wednesday) 23:00
# </div>
# 
# # Exercise 2. Recommender system
# 
# In this exercise, your task is to design a recommender system.
# 
# ## Learning goals:
# * Practise tuning a neural network model by using different regularization methods.

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import tools
import data


# In[32]:


skip_training = True  # Set this flag to True before validation and submission


# In[17]:


# During evaluation, this cell sets skip_training to True
# skip_training = True

import tools, warnings
warnings.showwarning = tools.customwarn


# In[5]:


# When running on your own computer, you can specify the data directory by:
# data_dir = tools.select_data_dir('/your/local/data/directory')
data_dir = tools.select_data_dir()


# In[6]:


# Select the device for training (use GPU if you have one)
#device = torch.device('cuda:0')
device = torch.device('cpu')


# In[7]:


if skip_training:
    # The models are always evaluated on CPU
    device = torch.device("cpu")


# ## Ratings dataset
# 
# We will train the recommender system on the dataset in which element consists of three values:
# * `user_id` - id of the user (the smallest user id is 1)
# * `item_id` - id of the movie (the smallest item id is 1)
# * `rating` - rating given by the user to the item (ratings are integer numbers between 1 and 5.
# 
# The recommender system need to predict the rating for any given pair of `user_id` and `item_id`.
# 
# We measure the quality of the predicted ratings using the mean-squared error (MSE) loss:
# $$
#   \frac{1}{N}\sum_{i=1}^N (r_i - \hat{r}_i)^2
# $$
# where $r_i$ is a real rating and $\hat{r}_i$ is a predicted one.
# 
# Note: The predicted rating $\hat{r}_i$ does not have to be an integer number.

# In[8]:


trainset = data.RatingsData(root=data_dir, train=True)
testset = data.RatingsData(root=data_dir, train=False)


# In[9]:


# Print one sample from the dataset
x = trainset[0]
print(f'user_id={x[0]}, item_id={x[1]}, rating={x[2]}')


# # Model
# 
# You need to design a recommender system model with the API described in the cell below.
# 
# Hints on the model architecture:
# * You need to use [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html?highlight=embedding#torch.nn.Embedding) layer to convert inputs `user_ids` and `item_ids` into reasonable representations. The idea of the embedding layer is that we want to represent similar users with values that are close to each other. The original representation as integers is not good for that. By using the embedding layer, we can learn such useful representations automatically.
# 
# ### Model tuning
# 
# In this exercise, you need to tune the architecture of your model to achieve the best performance on the provided test set. You will notice that overfitting is a severe problem for this data: The model can easily overfit the training set producing poor accuracy on the out-of-training (test) data.
# 
# You need to find an optimal combination of the hyperparameters, with some hyperparameters corresponding to the regularization techniques that we studied in the lecture.
# 
# The hyperparameters that you are advised to consider:
# * Learning rate value and learning rate schedule (decresing the learning rate often has positive effect on the model performance)
# * Number of training epochs
# * Network size
# * Weight decay
# * Early stopping
# * Dropout
# * Increase amount of data:
#   * Data augmentation
#   * Injecting noise
# 
# You can tune the hyperparameters by, for example, grid search, random search or manual tuning. In that case, you can use `architecture` argument to specify the hyperparameters that define the architecture of your network. After you have tuned the hyperparameters, set the default value of this argument to the optimal set of the hyparameters so that the best architecture is used in the accuracy tests.
# 
# Note:
# * The number of points that you will get from this exercise depends on the MSE loss on the test set:
#   * below 1.00: 1 point
#   * below 0.95: 2 points
#   * below 0.92: 3 points
#   * below 0.90: 4 points
#   * below 0.89: 5 points
#   * below 0.88: 6 points 

# In[19]:


class RecommenderSystem(nn.Module):
    def __init__(self, n_users, n_items,
                 architecture=None  
                 # If you want to tune the hyperparameters automatically (e.g. using random
                 # search), use this argument to specify the hyperparameters that define the
                 # architecture of your network. After you have tuned the hyperparameters,
                 # set the default value of this argument to the optimal set of the hyparameters
                 # so that the best architecture is used in the accuracy tests.
                ):
        """
        Args:
          n_users: Number of users.
          n_items: Number of items.
        """
        # YOUR CODE HERE
        super().__init__()
        self.Embed1 = nn.Embedding(n_users,25)
        self.Linea1 = nn.Linear(25, 30)
        self.Embed2 = nn.Embedding(n_items,25)
        self.Linea2 = nn.Linear(25, 30)
        
        self.tunning = nn.Sequential(
            nn.Linear(60, 50),
            nn.Tanh(),
            nn.Dropout(p=0.3), 
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(50, 1),
            nn.Tanh(),
        )
#         raise NotImplementedError()
        
    def forward(self, user_ids, item_ids):
        """
        Args:
          user_ids of shape (batch_size): User ids (starting from 1).
          item_ids of shape (batch_size): Item ids (starting from 1).
        
        Returns:
          outputs of shape (batch_size): Predictions of ratings.
        """
        # YOUR CODE HERE
        h1 = self.Linea1(self.Embed1(user_ids-1))
        h2 = self.Linea1(self.Embed2(item_ids-1))
        h3 = torch.cat([h1, h2], dim=1)
        y  = torch.flatten(self.tunning(h3)) #Rating is from 1-5
        return(y*6)
        raise NotImplementedError()


# You can test the shapes of the model outputs using the function below.

# In[10]:


def test_RecommenderSystem_shapes():
    n_users, n_items = 100, 1000
    model = RecommenderSystem(n_users, n_items)
    batch_size = 10
    user_ids = torch.arange(1, batch_size+1)
    item_ids = torch.arange(1, batch_size+1)
    output = model(user_ids, item_ids)
    print(output.shape)
    assert output.shape == torch.Size([batch_size]), "Wrong output shape."
    print('Success')

test_RecommenderSystem_shapes()


# In[50]:


# This cell is reserved for testing


# ## Train the model
# 
# You need to train a recommender system using **only the training data.** Please use the test set to select the best model: the model that generalizes best to out-of-training data.
# 
# **IMPORTANT**:
# * During testing, the predictions are produced by `predictions = model(user_ids, item_ids)` with the `user_ids` and `item_ids` loaded from `RatingsData`.
# * There is a size limit of 30Mb for saved models.

# In[24]:


trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)
print(trainset.n_users)
print(trainset.n_items)


# In[11]:


criterion = nn.MSELoss()
def compute_loss(model, testloader):
#     model.eval()

    with torch.no_grad():
        for user_ids, item_ids, rating in testloader:
            user_ids, item_ids, rating = user_ids.to(device), item_ids.to(device), rating.to(device)
            outputs = model.forward(user_ids,item_ids)
            rating = rating.float()
            loss = criterion(outputs, rating)
    return loss.cpu().numpy()

def print_progress(epoch, train_error, val_error):
    print('Epoch {}: Train error: {:.4f}, Test error: {:.4f}'.format(
        epoch, train_error, val_error))


# In[13]:


# Create the model
# IMPORTANT: the default value of the architecture argument should define your best model.
model = RecommenderSystem(trainset.n_users, trainset.n_items)


# In[29]:


learning_rate = 0.001
weight_decay = 0.001
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.MSELoss()

# Implement the training loop in this cell
if not skip_training:
    # YOUR CODE HERE
    n_epochs = 500
    train_errors = []
    val_errors = []

    for epoch in range(n_epochs):
#         model.train()
        for user_ids, item_ids, rating in trainloader:
            optimizer.zero_grad()
            outputs = model.forward(user_ids, item_ids)
            rating = rating.float()
            loss = criterion(outputs, rating)

            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            train_errors.append(compute_loss(model, trainloader))
            val_errors.append(compute_loss(model, testloader))
            print_progress(epoch, train_errors[-1], val_errors[-1])

#     raise NotImplementedError()


# In[30]:


# Save the model to disk (the pth-files will be submitted automatically together with your notebook)
# Set confirm=False if you do not want to be asked for confirmation before saving.
if not skip_training:
    tools.save_model(model, 'recsys.pth', confirm=True)


# In[33]:


# This cell loads your best model
if skip_training:
    model = RecommenderSystem(trainset.n_users, trainset.n_items)
    tools.load_model(model, 'recsys.pth', device)


# In[34]:


accuracy = compute_loss(model, testloader)
print('Accuracy of the network on the test images: %.3f' % accuracy)


# The next cell tests the accuracy of your best model. It is enough to submit .pth files.
# 
# **IMPORTANT**:
# * During testing, the predictions are produced by `predictions = model(user_ids, item_ids)` with the `user_ids` and `item_ids` loaded from `RatingsData`.
# * There is a size limit of 30Mb for saved models. Please make sure that your model loads in the cell above.

# In[ ]:


# This cell tests the accuracy of your best model.


# In[ ]:


# This cell is reserved for grading


# In[ ]:


# This cell is reserved for grading


# In[ ]:


# This cell is reserved for grading


# In[ ]:


# This cell is reserved for grading


# In[ ]:


# This cell is reserved for grading


# In[ ]:


# This cell is reserved for grading

