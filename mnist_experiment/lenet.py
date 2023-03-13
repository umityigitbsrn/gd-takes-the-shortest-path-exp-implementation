from torch import reshape
from torch import nn


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()

        self.num_classes = num_classes

        # self.conv_layer_01 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, bias=False),
        #     nn.Sigmoid(),
        #     nn.AvgPool2d(kernel_size=2, stride=2)
        # )
        #
        # self.conv_layer_02 = nn.Sequential(
        #     nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, bias=False),
        #     nn.Sigmoid(),
        #     nn.AvgPool2d(kernel_size=2, stride=2)
        # )
        #
        # self.fc_01 = nn.Linear(in_features=400, out_features=120, bias=False)
        # self.fc_02 = nn.Linear(in_features=120, out_features=84, bias=False)
        # self.fc_03 = nn.Linear(in_features=84, out_features=self.num_classes, bias=False)

        self.conv_layer_01 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.conv_layer_02 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.fc_01 = nn.Linear(in_features=400, out_features=120)
        self.fc_02 = nn.Linear(in_features=120, out_features=84)
        self.fc_03 = nn.Linear(in_features=84, out_features=self.num_classes)

    def forward(self, x):
        x = self.conv_layer_01(x)
        x = self.conv_layer_02(x)

        x = reshape(x, (x.shape[0], -1))
        x = self.fc_01(x)
        x = self.fc_02(x)
        out = self.fc_03(x)

        return out
