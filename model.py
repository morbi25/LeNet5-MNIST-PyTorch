from torch import nn


class LeNet5Mod(nn.Module):

    def __init__(self):
        super(LeNet5Mod, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, (5, 5)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 120, (5, 5)),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.model(x)
