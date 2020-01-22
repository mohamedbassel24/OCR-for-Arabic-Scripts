import torch
import torch.nn as nn
import torch.nn.functional as F
from commonfunctions import *


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(15, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 29)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

net.load_state_dict(torch.load("output3"))


def Predict_NN(X):
    Y = torch.from_numpy(np.array(X[0:15]))

    out = net(Y.float())
    return torch.argmax((out))
