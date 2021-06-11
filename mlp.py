"""
a simple multilayer perceptron
"""
import torch.nn as nn
import torch.nn.functional as F


class MLP1(nn.Module):
    def __init__(self, hidden_size, dropout_p, num_classes=10):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        #self.dropout3 = nn.Dropout2d(dropout_p)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        #x = self.dropout3(x)
        return x

class MLP2(nn.Module):
    def __init__(self, hidden_size, dropout_p, num_classes=10):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        #self.dropout3 = nn.Dropout2d(dropout_p)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = self.dropout3(x)
        return x
