#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-info">
# <b>Deadline:</b> March 29, 2023 (Wednesday) 23:00
# </div>
# 
# # Exercise 1. Sequence-to-sequence modeling with recurrent neural networks
# 
# The goals of this exercise are
# * to get familiar with recurrent neural networks used for sequential data processing
# * to get familiar with the sequence-to-sequence model for machine translation
# * to learn PyTorch tools for batch processing of sequences with varying lengths
# * to learn how to write a custom `DataLoader`
# 
# You may find it useful to look at this tutorial:
# * [Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

# In[1]:


skip_training = True  # Set this flag to True before validation and submission


# In[2]:


# During evaluation, this cell sets skip_training to True
# skip_training = True

import tools, warnings
warnings.showwarning = tools.customwarn


# In[3]:


import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

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


# ## Data
# 
# The dataset that we are going to use consists of pairs of sentences in French and English.

# In[7]:


from data import TranslationDataset, MAX_LENGTH, SOS_token, EOS_token

trainset = TranslationDataset(data_dir, train=True)


# * `TranslationDataset` supports indexing as required by `torch.utils.data.Dataset`.
# * Sentences are tensors of maximum length `MAX_LENGTH`.
# * Words in a (sentence) tensor are represented as an index (integer) in a language vocabulary.
# * The string representation of a word from the source language can be obtained from index `i` with `dataset.input_lang.index2word[i]`.
# * Similarly for the target language `dataset.output_lang.index2word[j]`.
# 
# Let us look at samples from that dataset.

# In[8]:


src_sentence, tgt_sentence = trainset[np.random.choice(len(trainset))]
print('Source sentence: "%s"' % ' '.join(trainset.input_lang.index2word[i.item()] for i in src_sentence))
print('Sentence as tensor of word indices:')
print(src_sentence)

print('Target sentence: "%s"' % ' '.join(trainset.output_lang.index2word[i.item()] for i in tgt_sentence))
print('Sentence as tensor of word indices:')
print(tgt_sentence)


# In[9]:


print('Number of source-target pairs in the training set: ', len(trainset))


# ## Custom DataLoader
# 
# We would like to train the sequence-to-sequence model using mini-batch training.
# One difficulty of mini-batch training in this case is that sequences may have varying lengths and this has to be taken into account when building the computational graph. Luckily, PyTorch has tools to support batch processing of such sequences.
# To use those tools, we need to write a custom data loader which puts sequences of varying lengths in the same tensor. We can customize the data loader by providing a custom `collate_fn` as explained [here](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
# 
# Our collate function:
# - combines sequences from the source language in a single tensor with extra values (at the end) filled with `PADDING_VALUE=0`.
# - combines sequences from the target language in a single tensor with extra values (at the end) filled with `PADDING_VALUE=0`.
# 
# **Important**:
# - Later in the code (not in this `collate` function), we will convert source sequences to objects of class [PackedSequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence) which can be processed by recurrent units such as `GRU` or `LSTM`. Conversion to `PackedSequence` objects requires sequences to be sorted by their lengths.
# **Therefore, the returned source sequences should be sorted by length in a decreasing order.**
# * The target sequences need not be sorted by their lengths because we have to keep the same order of sequences in the source and target tensors.
# 
# Your task is to implement the collate function.

# In[10]:


PADDING_VALUE = 0


# ## Sequence-to-sequence model for machine translation
# 
# In this exercise, we are going to build a machine translation system which transforms a sentence in one language into a sentence in another one. The computational graph of the translation model is shown below:
# 
# <img src="seq2seq.png" width=900>
# 
# We are going to use a simplified model without the dotted connections.

# In[11]:


from torch.nn.utils.rnn import pad_sequence

def collate(list_of_samples):
    """Merges a list of samples to form a mini-batch.

    Args:
      list_of_samples is a list of tuples (src_seq, tgt_seq):
          src_seq is of shape (src_seq_length,)
          tgt_seq is of shape (tgt_seq_length,)

    Returns:
      src_seqs of shape (max_src_seq_length, batch_size): Tensor of padded source sequences.
          The sequences should be sorted by length in a decreasing order, that is src_seqs[:,0] should be
          the longest sequence, and src_seqs[:,-1] should be the shortest.
      src_seq_lengths: List of lengths of source sequences.
      tgt_seqs of shape (max_tgt_seq_length, batch_size): Tensor of padded target sequences.
    """
    # YOUR CODE HERE
    # Get src_seq, tgt_seq, src_seq_length, batch_size.
    batch_size = len(list_of_samples)  
    src_seq, tgt_seq = [a for a, b in list_of_samples], [b for a, b in list_of_samples]
    src_seq_lengths = [torch.tensor(a.size()) for a, b in list_of_samples]

    # Sort src and tgt according to length
    src_len_sorted, idx = torch.sort(torch.as_tensor(src_seq_lengths),descending=True)
    src_len_sorted = list(src_len_sorted)
    src_seqs_sorted = [src_seq[i] for i in idx]
    tgt_seqs_sorted = [tgt_seq[i] for i in idx]
    
    # padding and transpose
    src_seqs_sorted = pad_sequence(src_seqs_sorted, batch_first=True)
    tgt_seqs_sorted = pad_sequence(tgt_seqs_sorted, batch_first=True)
    src_seqs_sorted = torch.transpose(src_seqs_sorted, 0, 1)
    tgt_seqs_sorted = torch.transpose(tgt_seqs_sorted, 0, 1)
    
    # Print and check result
#     print(src_seqs_sorted)
#     print(src_len_sorted)
#     print(tgt_seqs_sorted)
    
    return src_seqs_sorted, src_len_sorted, tgt_seqs_sorted
    raise NotImplementedError()


# In[12]:


# This cell tests collate() function

def test_collate_fn():
    pairs = [
        (torch.tensor([1, 2]), torch.tensor([3, 4, 5])),
        (torch.tensor([6, 7, 8]), torch.tensor([9, 10])),
        (torch.tensor([11, 12, 13, 14]), torch.tensor([15])),
    ]
    pad_src_seqs, src_seq_lengths, pad_tgt_seqs = collate(pairs)
    assert pad_src_seqs.shape == torch.Size([4, 3]), f"Bad pad_src_seqs.shape: {pad_src_seqs.shape}"
    assert pad_tgt_seqs.shape == torch.Size([3, 3]), f"Bad pad_tgt_seqs.shape: {pad_tgt_seqs.shape}"
    print('Source sequences combined:')
    print(pad_src_seqs)
    expected = torch.tensor([
      [11, 6, 1],
      [12, 7, 2],
      [13, 8, 0],
      [14, 0, 0],
    ])
    assert (pad_src_seqs == expected).all(), "pad_src_seqs does not match expected values"

    print(src_seq_lengths)
    if isinstance(src_seq_lengths[0], torch.Size):
        src_seq_lengths = sum((list(l) for l in src_seq_lengths), [])
    else:
        src_seq_lengths = [int(l) for l in src_seq_lengths]
    assert src_seq_lengths == [4, 3, 2], f"Bad src_seq_lengths: {src_seq_lengths}"

    print('Target sequences combined:')
    print(pad_tgt_seqs)
    expected = torch.tensor([
      [15,  9, 3],
      [ 0, 10, 4],
      [ 0,  0, 5],
    ])
    assert (pad_tgt_seqs == expected).all(), "pad_tgt_seqs0 does not match expected values"
    print('Success')

test_collate_fn()


# In[13]:


def test_collate_shapes():
    pairs = [
        (torch.LongTensor([1, 2]), torch.LongTensor([3, 4, 5])),
        (torch.LongTensor([6, 7, 8]), torch.LongTensor([9, 10])),
    ]
    pad_src_seqs, src_seq_lengths, pad_tgt_seqs = collate(pairs)
    assert type(src_seq_lengths) == list, "src_seq_lengths should be a list."
    assert pad_src_seqs.shape == torch.Size([3, 2]), f"Bad pad_src_seqs.shape: {pad_src_seqs.shape}"
    assert pad_src_seqs.dtype == torch.long
    assert pad_tgt_seqs.shape == torch.Size([3, 2]), f"Bad pad_tgt_seqs.shape: {pad_tgt_seqs.shape}"
    assert pad_tgt_seqs.dtype == torch.long
    print('Success')

test_collate_shapes()


# In[14]:


# We create custom DataLoader using the implemented collate function
# We are going to process 64 sequences at the same time (batch_size=64)
trainloader = DataLoader(dataset=trainset, batch_size=64, shuffle=True, collate_fn=collate, pin_memory=True)


# In[15]:


class Encoder(nn.Module):
    def __init__(self, src_dictionary_size, embed_size, hidden_size):
        """
        Args:
          src_dictionary_size: The number of words in the source dictionary.
          embed_size: The number of dimensions in the word embeddings.
          hidden_size: The number of features in the hidden state of GRU.
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(src_dictionary_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size)

    def forward(self, pad_seqs, seq_lengths, hidden):
        """
        Args:
          pad_seqs of shape (max_seq_length, batch_size): Padded source sequences.
          seq_lengths: List of sequence lengths.
          hidden of shape (1, batch_size, hidden_size): Initial states of the GRU.

        Returns:
          outputs of shape (max_seq_length, batch_size, hidden_size): Padded outputs of GRU at every step.
          hidden of shape (1, batch_size, hidden_size): Updated states of the GRU.
        """
        # YOUR CODE HERE
        embedded_seq  = self.embedding(pad_seqs)
        pack_in       = pack_padded_sequence(embedded_seq, seq_lengths)
        pack_out, hidden = self.gru(pack_in, hidden)
        
        seq_unpacked, lens_unpacked = pad_packed_sequence(pack_out)
        
        return seq_unpacked, hidden
        raise NotImplementedError()

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size)


# In[16]:


def test_Encoder_shapes():
    hidden_size = 3
    encoder = Encoder(src_dictionary_size=5, embed_size=10, hidden_size=hidden_size)

    max_seq_length = 4
    batch_size = 2
    hidden = encoder.init_hidden(batch_size=batch_size)
    pad_seqs = torch.tensor([
        [        1,             2],
        [        2,     EOS_token],
        [        3, PADDING_VALUE],
        [EOS_token, PADDING_VALUE]
    ])

    outputs, new_hidden = encoder.forward(pad_seqs=pad_seqs, seq_lengths=[4, 2], hidden=hidden)
    assert outputs.shape == torch.Size([4, batch_size, hidden_size]), f"Bad outputs.shape: {outputs.shape}"
    assert new_hidden.shape == torch.Size([1, batch_size, hidden_size]), f"Bad new_hidden.shape: {new_hidden.shape}"
    print('Success')

test_Encoder_shapes()


# ## Encoder
# 
# The encoder encodes a source sequence $(x_1, x_2, ..., x_T)$ into a single vector $h_T$ using the following recursion:
# $$
#   h_{t} = f(h_{t-1}, x_t) \qquad t = 1, \ldots, T
# $$
# where:
# * intial state $h_0$ is often chosen arbitrarily (we choose it to be zero)
# * function $f$ is defined by the type of the RNN cell (in our experiments, we will use [GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU))
# * $x_t$ is a vector that represents the $t$-th word in the source sentence.
# 
# A common practice in natural language processing is to _learn_ the word representations $x_t$ (instead of, for example, using one-hot coded vectors). In PyTorch, this is supported by class [Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding) which we are going to use.
# 
# The computational graph of the encoder is shown below:
# 
# <img src="seq2seq_encoder.png" width=500>
# 
# Your task is to implement the `forward` function of the encoder. It should contain the following steps:
# * Embed the words of the source sequences.
# * Pack source sequences using [`pack_padded_sequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence). This converts padded source sequences into an object that can be processed by PyTorch recurrent units such as `nn.GRU` or `nn.LSTM`.
# * Apply GRU computations to packed sequences obtained in the previous step
# * Convert packed sequence of GRU outputs into padded representation with [`pad_packed_sequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html#torch.nn.utils.rnn.pad_packed_sequence).

# In[17]:


# This cell tests Encoder
import tests

def test_Encoder(Encoder):
    with torch.no_grad():
        net = Encoder(src_dictionary_size=5, embed_size=2, hidden_size=2)
        tests.set_weights_encoder(net)

        pad_seqs = torch.tensor([
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 0]
        ])  # (max_seq_length, batch_size)
        seq_lengths = [4, 2]
        hidden = net.init_hidden(batch_size=2)

        outputs, new_hidden = net.forward(pad_seqs, seq_lengths, hidden)

        expected = torch.tensor([
            [ 0.0000, -0.0150],
            [ 0.0004, -0.0221],
            [ 0.0007, -0.0055],
            [ 0.0005,  0.0323]
        ])
        print('outputs[:, 0, :]:\n', outputs[:, 0, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,0,:], expected, atol=1e-4), "outputs do not match expected values"

        expected = torch.tensor([
            [ 0.0000, -0.0150],
            [ 0.0004, -0.0021]
        ])
        print('outputs[:2, 1, :]:\n', outputs[:2, 1, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:2,1,:], expected, atol=1e-4), "outputs do not match expected values"

        expected = torch.tensor([[
            [ 0.0005,  0.0323],
            [ 0.0004, -0.0021]
        ]])
        print('new_hidden:\n', new_hidden)
        print('expected:\n', expected)
        assert torch.allclose(new_hidden, expected, atol=1e-4), "new_hidden does not match expected value"
        print('Success')

test_Encoder(Encoder)


# ## Decoder
# 
# The decoder takes as input the representation computed by the encoder and transforms it into a sentence in the target language. The computational graph of the decoder is shown below:
# 
# <img src="seq2seq_decoder.png" width=500 align="top">
# 
# * $z_0$ is the output of the encoder, that is $z_0 = h_5$, thus `hidden_size` of the decoder should be the same as `hidden_size` of the encoder.
# * $y_{i}$ are the log-probabilities of the words in the target language, the dimensionality of $y_{i}$ is the size of the target dictionary.
# * $z_{i}$ is mapped to $y_{i}$ using a linear layer `self.out` followed by `F.log_softmax` (because we use `nn.NLLLoss` loss for training).
# * Each cell of the decoder is a GRU, it receives as inputs the previous state $z_{i-1}$ and relu of the **embedding** of the previous word. Thus, you need to embed the words of the target language as well. The previous word is taken as the word with the maximum log-probability.
# 
# Note that the decoder outputs a word at every step and the same word is used as the input to the recurrent unit at the next step. At the beginning of decoding, the previous word input is fed with a special word SOS which stands for "start of a sentence". During training, we know the target sentence for decoding, therefore we can feed the correct words $y_i$ as inputs to the recurrent unit.
# 
# There is one extra thing that it is wise to take care of. When the target sentence is fed to the decoder during training, the decoder learns to generate only the next word (this scenario is called "teacher forcing"). At test time, the decoder works differently: it generates the whole sequence using its own predictions as inputs at each step. Therefore, it makes sense to train the decoder to produce full sentences. In order to do that, we will alternate between two modes during training:
# * "teacher forcing": the decoder is fed with the words in the target sequence
# * no "teacher forcing": the decoder generates the output sequence using its own predictions. In this case, we will generate sequences of the same length as the length of the longest sequence in `pad_tgt_seqs` (if `pad_tgt_seqs` is not `None`) or of length `MAX_LENGTH` (if `pad_tgt_seqs` is `None`).
# 
# You need to implement the decoder which has the structure shown in the figure above.
# 
# Notes:
# * `SOS_token` is imported at the beginning of the notebook.
# * **Running this code on GPU sometimes fails producing a CUDA error (if you know the reason, please let us know).** If this happens to you, please train the model on CPU.

# In[18]:


class Decoder(nn.Module):
    def __init__(self, tgt_dictionary_size, embed_size, hidden_size):
        """
        Args:
          tgt_dictionary_size: The number of words in the target dictionary.
          embed_size: The number of dimensions in the word embeddings.
          hidden_size: The number of features in the hidden state.
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_dictionary_size = tgt_dictionary_size

        self.embedding = nn.Embedding(tgt_dictionary_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, tgt_dictionary_size)

    def forward(self, hidden, pad_tgt_seqs=None, teacher_forcing=False):
        """
        Args:
          hidden of shape (1, batch_size, hidden_size): States of the GRU.
          pad_tgt_seqs of shape (max_out_seq_length, batch_size): Tensor of words (word indices) of the
              target sentence. If None, the output sequence is generated by feeding the decoder's outputs
              (teacher_forcing has to be False).
          teacher_forcing (bool): Whether to use teacher forcing or not.

        Returns:
          outputs of shape (max_out_seq_length, batch_size, tgt_dictionary_size): Tensor of log-probabilities
              of words in the target language.
          hidden of shape (1, batch_size, hidden_size): New states of the GRU.

        Note: Do not forget to transfer tensors that you may want to create in this function to the device
        specified by `hidden.device`.
        """
        if pad_tgt_seqs is None:
            assert not teacher_forcing, 'Cannot use teacher forcing without a target sequence.'

        # YOUR CODE HERE
        batch_size = hidden.size(1)
        target_length = pad_tgt_seqs.size(0) if pad_tgt_seqs is not None else MAX_LENGTH
        output = torch.zeros(target_length, batch_size, self.tgt_dictionary_size)
        
        y_input = torch.ones((1, batch_size), dtype=torch.long).to(hidden.device) * SOS_token
        z_input = hidden
        
        for di in range(target_length):
            Embed_Input = F.relu(self.embedding(y_input))
            z_cur, hidden = self.gru(Embed_Input,z_input)
            y_cur = F.log_softmax(self.out(z_cur[0]), dim=1)
#             print(di, y_input.shape, z_input.shape, Embed_Input.shape, z_cur.shape)
            
            z_input = hidden
            if teacher_forcing is True:
                y_input = pad_tgt_seqs[di].unsqueeze(0)
            else:
                topv, topi = y_cur.topk(1)
                y_input = topi.squeeze().detach().unsqueeze(0)
                    
            output[di] =  y_cur
        
        return output, hidden
        raise NotImplementedError()


# In[19]:


def test_Decoder_shapes():
    hidden_size = 2
    tgt_dictionary_size = 5
    test_decoder = Decoder(tgt_dictionary_size, embed_size=10, hidden_size=hidden_size)

    max_seq_length = 4
    batch_size = 2
    pad_tgt_seqs = torch.tensor([
        [        1,             2],
        [        2,     EOS_token],
        [        3, PADDING_VALUE],
        [EOS_token, PADDING_VALUE]
    ])  # [max_seq_length, batch_size]

    hidden = torch.zeros(1, batch_size, hidden_size)
    outputs, new_hidden = test_decoder.forward(hidden, pad_tgt_seqs, teacher_forcing=False)

    assert outputs.size(0) <= 4, f"Too long output sequence: outputs.size(0)={outputs.size(0)}"
    assert outputs.shape[1:] == torch.Size([batch_size, tgt_dictionary_size]), \
        f"Bad outputs.shape[1:]={outputs.shape[1:]}"
    assert new_hidden.shape == torch.Size([1, batch_size, hidden_size]), f"Bad new_hidden.shape={new_hidden.shape}"

    outputs, new_hidden = test_decoder.forward(hidden, pad_tgt_seqs, teacher_forcing=True)
    assert outputs.shape == torch.Size([4, batch_size, tgt_dictionary_size]), \
        f"Bad shape outputs.shape={outputs.shape}"
    assert new_hidden.shape == torch.Size([1, batch_size, hidden_size]), f"Bad new_hidden.shape={new_hidden.shape}"

    # Generation mode
    outputs, new_hidden = test_decoder.forward(hidden, None, teacher_forcing=False)
    assert outputs.shape[1:] == torch.Size([batch_size, tgt_dictionary_size]), \
        f"Bad outputs.shape[1:]={outputs.shape[1:]}"
    assert new_hidden.shape == torch.Size([1, batch_size, hidden_size]), f"Bad new_hidden.shape={new_hidden.shape}"

    print('Success')

test_Decoder_shapes()


# In[20]:


# This cell tests Decoder
def test_Decoder_no_forcing(Decoder):
    # Test without teaching_forcing
    with torch.no_grad():
        net = Decoder(tgt_dictionary_size=5, embed_size=2, hidden_size=2)
        tests.set_weights_decoder(net)

        pad_target_seqs = torch.tensor([
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 0]
        ])

        hidden = torch.tensor([
            [1., -1.],
            [1., -1.],
        ]).view(1, 2, 2)
        outputs, new_hidden = net.forward(hidden, pad_target_seqs, teacher_forcing=False)
        outputs, new_hidden = outputs.float(), new_hidden.float()

        expected = torch.tensor([
            [-1.3788, -1.3503, -1.5045, -2.1493, -1.8954],
            [-1.5265, -1.4880, -1.5090, -1.8655, -1.7096],
            [-1.6097, -1.5715, -1.5319, -1.7189, -1.6249],
            [-1.6317, -1.6037, -1.5593, -1.6543, -1.6007]
        ])

        print('outputs[:, 0, :]:\n', outputs[:, 0, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,0,:], expected, atol=1e-4), "outputs do not match expected values"

        print('outputs[:, 1, :]:\n', outputs[:, 1, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,1,:], expected, atol=1e-4), "outputs do not match expected values"

        print('Success')

test_Decoder_no_forcing(Decoder)


# In[21]:


# This cell tests Decoder
def test_Decoder_with_forcing(Decoder):
    # Test with teaching_forcing
    with torch.no_grad():
        net = Decoder(tgt_dictionary_size=5, embed_size=2, hidden_size=2)
        tests.set_weights_decoder(net)

        pad_target_seqs = torch.tensor([
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 0]
        ])

        hidden = torch.tensor([
            [1., -1.],
            [1., -1.],
        ]).view(1, 2, 2)
        outputs, new_hidden = net.forward(hidden, pad_target_seqs, teacher_forcing=True)
        outputs, new_hidden = outputs.float(), new_hidden.float()

        expected = torch.tensor([
            [-1.3788, -1.3503, -1.5045, -2.1493, -1.8954],
            [-1.5265, -1.4880, -1.5090, -1.8655, -1.7096],
            [-1.5906, -1.5591, -1.5397, -1.7299, -1.6393],
            [-1.6109, -1.5899, -1.5668, -1.6664, -1.6159]
        ])
        print('outputs[:, 0, :]:\n', outputs[:, 0, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,0,:], expected, atol=1e-4), "outputs do not match expected values"

        expected = torch.tensor([
            [-1.3788, -1.3503, -1.5045, -2.1493, -1.8954],
            [-1.5091, -1.4770, -1.5174, -1.8772, -1.7245],
            [-1.5706, -1.5460, -1.5476, -1.7424, -1.6548],
            [-1.6209, -1.5961, -1.5623, -1.6618, -1.6087]
        ])
        print('outputs[:, 1, :]:\n', outputs[:, 1, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,1,:], expected, atol=1e-4), "outputs do not match expected values"

        print('Success')

test_Decoder_with_forcing(Decoder)


# In[22]:


# This cell tests Decoder
def test_Decoder_generation(Decoder):
    # Test in generation mode
    with torch.no_grad():
        net = Decoder(tgt_dictionary_size=5, embed_size=2, hidden_size=2)
        tests.set_weights_decoder(net)

        hidden = torch.tensor([
            [1., -1.],
            [1., -1.],
        ]).view(1, 2, 2)
        outputs, new_hidden = net.forward(hidden, None, teacher_forcing=False)
        outputs, new_hidden = outputs.float(), new_hidden.float()

        expected = torch.tensor([
            [-1.3788, -1.3503, -1.5045, -2.1493, -1.8954],
            [-1.5265, -1.4880, -1.5090, -1.8655, -1.7096],
            [-1.6097, -1.5715, -1.5319, -1.7189, -1.6249],
            [-1.6317, -1.6037, -1.5593, -1.6543, -1.6007],
            [-1.6417, -1.6201, -1.5757, -1.6212, -1.5899],
            [-1.6459, -1.6282, -1.5851, -1.6042, -1.5852],
            [-1.6476, -1.6321, -1.5904, -1.5956, -1.5831],
            [-1.6541, -1.6379, -1.5912, -1.5881, -1.5782],
            [-1.6571, -1.6408, -1.5919, -1.5842, -1.5759],
            [-1.6585, -1.6421, -1.5924, -1.5822, -1.5748]
        ])
        print('outputs[:, 0, :]:\n', outputs[:, 0, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,0,:], expected, atol=1e-4), "outputs do not match expected values"

        print('outputs[:, 1, :]:\n', outputs[:, 1, :])
        print('expected:\n', expected)
        assert torch.allclose(outputs[:,1,:], expected, atol=1e-4), "outputs do not match expected values"

        print('Success')

test_Decoder_generation(Decoder)


# ## Training of sequence-to-sequence model using mini-batches
# 
# Now we are going to train the sequence-to-sequence model on the toy translation dataset.

# In[23]:


# Create the seq2seq model
hidden_size = embed_size = 256
encoder = Encoder(trainset.input_lang.n_words, embed_size, hidden_size).to(device)
decoder = Decoder(trainset.output_lang.n_words, embed_size, hidden_size).to(device)


# In[24]:


teacher_forcing_ratio = 0.5


# Implement the training loop in the cell below. In the training loop, we first encode source sequences using the encoder, then we decode the encoded state using the decoder. The decoder outputs log-probabilities of words in the target language. We need to use these log-probabilities and the indexes of the words in the target sequences to compute the loss.
# 
# The loss is
# \begin{align*}
# L = - \frac{1}{N} \sum_{n} \sum_{t=1}^{T_n}
# \log p\left(\mathbf{y}_t^{(n)} \:\Bigl|\: \mathbf{y}_{<t}^{(n)}, \mathbf{X}^{(n)} \right)
# \end{align*}
# where $T_n$ is the length of the $n$-th target sequence and $N= \sum_{n=1} T_n$ is the total number of words in all the sentences of the mini-batch.
# 
# Recommended hyperparameters:
# - Encoder optimizer: Adam with learning rate 0.001
# - Decoder optimizer: Adam with learning rate 0.001
# - Number of epochs: 30
# - Toggle `teacher_forcing` on and off (for each mini-batch) according to the `teacher_forcing_ratio` specified above.
# 
# Hints:
# - Training should proceed relatively fast.
# - If you do well, the training loss should reach 0.1 in 30 epochs.
# - Slight overlearning may happen (you can see that if you track the test error during training) but you can ignore this problem. 
# - **Important:** When computing the loss, you need to ignore the padded values. This can easily be done by using argument `ignore_index` of function [`nll_loss`](
# https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.nll_loss).

# In[25]:


n_epochs = 30
learning_rate = 0.001
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss(ignore_index = PADDING_VALUE)

if not skip_training:
    # YOUR CODE HERE

    for epoch in range(n_epochs):
        for src_seq,src_seq_length,tgt_seq in trainloader:
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            encoder_hidden = encoder.init_hidden(batch_size = src_seq.size(1))
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(src_seq,src_seq_length, encoder_hidden)
            decoder_hidden = encoder_hidden
            decoder_output, decoder_hidden = decoder(decoder_hidden, tgt_seq, use_teacher_forcing)
            
            tgt_seq = torch.flatten(tgt_seq, end_dim = 1)
            decoder_output = torch.flatten(decoder_output, end_dim = 1)
            
#             print(decoder_output.shape)
#             print(target_tensor.shape)
            
            loss = criterion(decoder_output, tgt_seq)
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
        
        print(epoch, loss)
    print("Trainning done")
    
#     raise NotImplementedError()


# In[39]:


# Save the model to disk (the pth-files will be submitted automatically together with your notebook)
# Set confirm=False if you do not want to be asked for confirmation before saving.
if not skip_training:
    tools.save_model(encoder, '1_rnn_encoder.pth', confirm=True)
    tools.save_model(decoder, '1_rnn_decoder.pth', confirm=True)


# In[27]:


if skip_training:
    hidden_size = 256
    encoder = Encoder(trainset.input_lang.n_words, embed_size, hidden_size)
    tools.load_model(encoder, '1_rnn_encoder.pth', device)
    
    decoder = Decoder(trainset.output_lang.n_words, embed_size, hidden_size)
    tools.load_model(decoder, '1_rnn_decoder.pth', device)


# In[28]:


# This cell tests training accuracy


# ## Evaluation
# 
# Next we need to implement a function that converts input sequences to output sequences using the trained sequence-to-sequence model.
# 
# Notes:
# * Since we do not need to compute the gradients in the evaluation phase, we can speed up the computations by using the statement `with torch.no_grad():`.
# * Please transfer the tensors to `device` inside this function.

# In[29]:


def translate(encoder, decoder, pad_src_seqs, src_seq_lengths):
    """Translate sequences from the source language to the target language using the trained model.
    
    Args:
      encoder (Encoder): Trained encoder.
      decoder (Decoder): Trained decoder.
      pad_src_seqs of shape (max_src_seq_length, batch_size): Padded source sequences.
      src_seq_lengths: List of source sequence lengths.
    
    Returns:
      out_seqs of shape (MAX_LENGTH, batch_size): LongTensor of word indices of the output sequences.
    """
    # YOUR CODE HERE
    with torch.no_grad():
        encoder_hidden = encoder.init_hidden(batch_size = pad_src_seqs.size(1))
        
        encoder_outputs, encoder_hidden = encoder(pad_src_seqs,src_seq_lengths, encoder_hidden)
        decoder_hidden = encoder_hidden
        decoder_output, decoder_hidden = decoder(decoder_hidden, teacher_forcing = False)
        
#         decoder_output = torch.flatten(decoder_output, end_dim = 1)
        topv, topi = decoder_output.topk(1)
        decoder_output = topi.squeeze().detach()
        
    return decoder_output
    raise NotImplementedError()


# In[30]:


def test_translate_shapes():
    pad_src_seqs = torch.tensor([
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 0]
    ])

    out_seqs = translate(encoder, decoder, pad_src_seqs, src_seq_lengths=[4, 2])
    assert out_seqs.shape == torch.Size([MAX_LENGTH, 2]), f"Wrong out_seqs.shape: {out_seqs.shape}"
    print('Success')

test_translate_shapes()


# Let us now translate a few sentences from the training set and print the source, target, and produced output.
# 
# If you trained the model well enough, the model should memorize the training data well.

# In[31]:


def seq_to_tokens(seq, lang):
    'Convert a sequence of word indices into a list of words (strings).'
    sentence = []
    for i in seq:
        if i == EOS_token:
            break
        sentence.append(lang.index2word[i.item()])
    return(sentence)

def seq_to_string(seq, lang):
    'Convert a sequence of word indices into a sentence string.'
    return(' '.join(seq_to_tokens(seq, lang)))


# In[32]:


# Translate a few sentences from the training set
print('Translate training data:')
print('-----------------------------')
pad_src_seqs, src_seq_lengths, pad_tgt_seqs = next(iter(trainloader))
out_seqs = translate(encoder, decoder, pad_src_seqs, src_seq_lengths)

for i in range(5):
    print('SRC:', seq_to_string(pad_src_seqs[:,i], trainset.input_lang))
    print('TGT:', seq_to_string(pad_tgt_seqs[:,i], trainset.output_lang))
    print('OUT:', seq_to_string(out_seqs[:,i], trainset.output_lang))
    print('')


# Now we translate random sentences from the test set. A well-trained model should output sentences that look similar to the target ones. The mistakes are usually done for words that were rare in the training set.

# In[33]:


testset = TranslationDataset(data_dir, train=False)
testloader = DataLoader(dataset=testset, batch_size=64, shuffle=False, collate_fn=collate)


# In[34]:


print('Translate test data:')
print('-----------------------------')
pad_src_seqs, src_seq_lengths, pad_tgt_seqs = next(iter(testloader))
out_seqs = translate(encoder, decoder, pad_src_seqs, src_seq_lengths)

for i in range(5):
    print('SRC:', seq_to_string(pad_src_seqs[:,i], testset.input_lang))
    print('TGT:', seq_to_string(pad_tgt_seqs[:,i], testset.output_lang))
    print('OUT:', seq_to_string(out_seqs[:,i], testset.output_lang))
    print('')


# ## Compute BLEU score
# 
# Let us now compute the [BLEU score](https://en.wikipedia.org/wiki/BLEU) for the translations produced by our model. We can use the PyTorch function [bleu_score](https://pytorch.org/text/stable/data_metrics.html?highlight=bleu_score#torchtext.data.metrics.bleu_score) for that.
# 
# * **Your model should achieve a minimum BLEU score of 90 on the training set.**
# * The BLEU score on the test set should be greater than 40.
# 
# The model can severly overfit to the training set and we do not cope with the overfitting problem in this exercise.

# In[35]:


from torchtext.data.metrics import bleu_score


# In[36]:


# Create translations for the training set
candidate_corpus = []
references_corpus = []
for pad_src_seqs, src_seq_lengths, pad_tgt_seqs in trainloader:
    out_seqs = translate(encoder, decoder, pad_src_seqs, src_seq_lengths)
    candidate_corpus.extend([seq_to_tokens(seq, trainset.output_lang) for seq in out_seqs.T])
    references_corpus.extend([[seq_to_tokens(seq, trainset.output_lang)] for seq in pad_tgt_seqs.T])

# Compute BLEU for translations
score = bleu_score(candidate_corpus, references_corpus)
print(f'BLEU score on training data: {score*100}')
assert score*100 > 90, "The BLEU score is too low."


# In[37]:


# Create translations for the test set
candidate_corpus = []
references_corpus = []
for pad_src_seqs, src_seq_lengths, pad_tgt_seqs in testloader:
    out_seqs = translate(encoder, decoder, pad_src_seqs, src_seq_lengths)
    candidate_corpus.extend([seq_to_tokens(seq, testset.output_lang) for seq in out_seqs.T])
    references_corpus.extend([[seq_to_tokens(seq, testset.output_lang)] for seq in pad_tgt_seqs.T])

# Compute BLEU for translations
score = bleu_score(candidate_corpus, references_corpus)
print(f'BLEU score on test data: {score*100}')
assert score*100 > 40, "The BLEU score is too low."


# <div class="alert alert-block alert-info">
# <b>Conclusion</b>
# </div>
# 
# In this notebook:
# * We learned how recurrent neural networks can be used to build a sequence-to-sequence model.
# * We trained a sequence-to-sequence model for statistical machine translation.
