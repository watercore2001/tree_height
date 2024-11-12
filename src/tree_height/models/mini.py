from torch import nn


class MiniModel(nn.Module):
    def __init__(self, in_feature: int):
        super().__init__()
        self.linear = nn.Conv2d(in_feature, 1, kernel_size=1)

    def forward(self, x):
        """
        :param x: b c h w
        :return: b 1 h w
        """
        return self.linear(x)

