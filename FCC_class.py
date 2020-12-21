import torch.nn as nn
import torch.nn.functional as F

class FCC(nn.Module):

    # Constructor
    def __init__(self):
        super(FCC, self).__init__()

        self.fLayer1 = nn.Linear(
            28*28,  # input image 28x28
            300,
            bias=True
        )

        self.fLayer2 = nn.Linear(
            300,  # input image 28x28
            300,
            bias=True
        )

        self.fLayer3 = nn.Linear(
            300,  # input image 28x28
            300,
            bias=True
        )

        self.fLayer4 = nn.Linear(
            300,  # input image 28x28
            300,
            bias=True
        )

        self.fLayer5 = nn.Linear(
            300,
            10,  # output layer for classification
            bias=True
        )

    # Foward function
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fLayer1(x))
        x = F.relu(self.fLayer2(x))
        x = F.relu(self.fLayer3(x))
        x = F.relu(self.fLayer4(x))
        x = F.log_softmax(self.fLayer5(x), dim=1)
        return x
