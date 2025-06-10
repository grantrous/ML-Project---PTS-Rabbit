import torch
import torch.nn as nn
class Model_V1(nn.Module):
    torch.manual_seed(36)  # For reproducibility
    def __init__(self):
        super(Model_V1, self).__init__()
        self.fc1 = nn.Linear(5, 10)  # Input layer to hidden layer
        self.fc2 = nn.Linear(10, 10) #Hidden layer to hidden layer
        self.fc3 = nn.Linear(10, 10) #Hidden layer to hidden layer
        self.fc4 = nn.Linear(10,3)   # Hidden layer to output layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        return self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))))
    