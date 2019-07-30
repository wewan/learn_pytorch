"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch import nn
import torch.nn.functional as F
import torch
import random




class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()
        if not isinstance(n_hidden,list):
            raise TypeError

        self.lth = len(n_hidden) + 1
        n_hidden.append(n_classes)
        temp = n_inputs

        self.linears = nn.ModuleList()
        for i in range(self.lth):
            self.linears.append(nn.Linear(temp,n_hidden[i]))
            temp = n_hidden[i]

        self.relu = nn.ReLU()
        self.softmax = F.softmax

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        x = x.contiguous().view(x.size(0), -1)
        for i in range(self.lth):
            x= self.linears[i](x)
            if i != self.lth - 1:
                x = self.relu(x)
            else:
                x = self.softmax(x)
        out = x
        ########################
        # END OF YOUR CODE    #
        #######################


        return out

if __name__ == '__main__':
    mlp_model = MLP(100,[10],10)
    print(i for i in mlp_model.parameters())
    print(mlp_model)
