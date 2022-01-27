import torch.nn as nn


# noinspection PyTypeChecker
class EncoderBlock(nn.Sequential):
    def __init__(self, input_channel, output_channel, kernel_size=3):
        padding = kernel_size // 2
        super().__init__(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # ToDo change to AvgPool
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, padding=padding, padding_mode="reflect"),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size=kernel_size, padding=padding, padding_mode="reflect"),
        )
