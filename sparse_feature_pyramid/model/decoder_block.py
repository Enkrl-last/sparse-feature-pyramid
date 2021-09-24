import torch.nn as nn
import torch


# noinspection PyTypeChecker,PyMethodOverriding
class DecoderBlock(nn.Module):
    def __init__(self, input_channel, skip_channel, output_channel, kernel_size=5, padding=None, stride=1, dilation=1):
        super().__init__()
        padding = padding or (dilation * (kernel_size - 1) // 2)
        self._convolution = nn.Sequential(
            nn.Conv2d(2 * skip_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                      padding_mode="reflect", dilation=dilation),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                      padding_mode="reflect", dilation=dilation),
            nn.BatchNorm2d(output_channel),
            nn.ReLU()
        )
        self._upsample_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(input_channel, skip_channel, kernel_size=kernel_size, padding=padding, padding_mode="reflect",
                      dilation=dilation)
        )

    def forward(self, x, skip_feature):
        x = self._upsample_conv(x)
        x = torch.cat([x, skip_feature], dim=1)
        return self._convolution(x)
