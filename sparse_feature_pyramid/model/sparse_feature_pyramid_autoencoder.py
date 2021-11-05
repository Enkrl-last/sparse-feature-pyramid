from .base_lightning_module import BaseLightningModule
from .encoder_block import EncoderBlock
from .decoder_block import DecoderBlock
import torch.nn as nn
import torch.nn.functional
import torch
import numpy as np
from kapture_localization.matching.matching import MatchPairNnTorch


def mean(x):
    return sum(x) / len(x)


# noinspection PyTypeChecker
class SparseFeaturePyramidAutoencoder(BaseLightningModule):
    def __init__(self, parameters):
        super().__init__(parameters)
        self._encoder_blocks = self.make_encoder_blocks(parameters.feature_dimensions)
        self._decoder_blocks = self.make_decoder_blocks(parameters.feature_dimensions)
        self._input_convolution = self.make_input_convolution(parameters.input_dimension,
                                                              parameters.feature_dimensions[0])
        self._output_convolutions = self.make_output_convolutions(parameters.feature_dimensions,
                                                                  parameters.input_dimension)
        self._mask_convolutions = self.make_mask_convolutions(parameters.feature_dimensions)
        self._loss = nn.MSELoss()

    @staticmethod
    def make_input_convolution(input_dimensions, output_dimension):
        return nn.Sequential(
            nn.Conv2d(input_dimensions, output_dimension, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(output_dimension),
            nn.ReLU(),
            nn.Conv2d(output_dimension, output_dimension, kernel_size=3, padding=1, padding_mode="reflect"),
        )

    @staticmethod
    def make_encoder_blocks(feature_dimensions):
        input_dimension = feature_dimensions[0]
        blocks = []
        for feature_dimension in feature_dimensions[1:]:
            output_dimension = feature_dimension
            blocks.append(EncoderBlock(input_dimension, output_dimension))
            input_dimension = feature_dimension
        return nn.ModuleList(blocks)

    @staticmethod
    def make_decoder_blocks(feature_dimensions):
        input_dimension = feature_dimensions[-1]
        blocks = []
        for feature_dimension in reversed(feature_dimensions[:-1]):
            output_dimension = feature_dimension
            blocks.append(DecoderBlock(input_dimension, output_dimension, output_dimension))
            input_dimension = feature_dimension
        return nn.ModuleList(blocks)

    @staticmethod
    def make_mask_convolutions(feature_dimensions):
        convolutions = []
        for feature_dimension in feature_dimensions:
            convolutions.append(nn.Conv2d(feature_dimension, 1, kernel_size=3, padding=1, padding_mode="reflect"))
        return nn.ModuleList(convolutions)

    @staticmethod
    def make_output_convolutions(feature_dimensions, input_dimension):
        convolutions = []
        for feature_dimension in feature_dimensions:
            convolutions.append(nn.Conv2d(feature_dimension, input_dimension, kernel_size=3, padding=1,
                                          padding_mode="reflect"))
        return nn.ModuleList(convolutions)

    def forward(self, x):
        x = self._input_convolution(x)
        feature_pyramid = [x]
        for block in self._encoder_blocks:
            x = block(x)
            feature_pyramid.append(x)

        masks = []
        probabilities = []
        kl_losses = []
        for feature, convolution in zip(feature_pyramid, self._mask_convolutions):
            log_prob = convolution(feature)
            mask, probability = self.predict_mask(log_prob)
            masks.append(mask)
            probabilities.append(probability)
            kl_losses.append(self.hparams.kl_loss_coefficient * self.predict_kl_loss(log_prob))
        masked_feature_pyramid = [feature * mask for feature, mask in zip(feature_pyramid, masks)]

        x = masked_feature_pyramid[-1]
        outputs = [self._output_convolutions[-1](x)]
        for i in range(len(self._decoder_blocks)):
            x = self._decoder_blocks[i](x, masked_feature_pyramid[-i - 2])
            outputs.append(self._output_convolutions[-i - 2](x))
        return list(reversed(outputs)), feature_pyramid, masks, probabilities, kl_losses

    def predict_mask(self, x):
        probability = torch.sigmoid(x)
        if self.training:
            a = torch.bernoulli(probability).detach()
            c = (a - probability).detach()
            mask = c + probability
        else:
            mask = torch.bernoulli(probability).detach()
            # mask = torch.where(probability > 0.5, 1., 0.)
        return mask, probability

    @staticmethod
    def predict_kl_loss(x):
        zeros = torch.zeros_like(x)
        stacked_x = torch.stack([x, zeros], dim=0)
        log_sum_exp = torch.logsumexp(stacked_x, dim=0)
        probability = torch.sigmoid(x)
        return x * probability - log_sum_exp + np.log(2)

    def loss(self, batch):
        input_image = batch["image"]
        mask = batch["mask"]
        output = self.forward(input_image)
        image_losses = [self.scaled_image_loss(input_image, output_image, mask) for output_image in output[0]]
        image_loss = mean(image_losses)
        size_loss = self.size_loss(output[3])
        loss = image_loss + self.hparams.size_loss_koef * size_loss  # ToDo
        kl_loss = mean([torch.mean(x) for x in output[4]])
        return output, {
            "loss": loss,
            "image_loss": image_loss,
            "image_loss4": image_losses[4],
            "size_loss": size_loss,
            "kl_loss": kl_loss
        }

    def scaled_image_loss(self, input_image, output_image, mask):
        scaled_input_image = torch.nn.functional.interpolate(input_image, size=output_image.shape[2:])
        scaled_mask = torch.nn.functional.interpolate(mask[:, None], size=output_image.shape[2:])
        return self._loss(scaled_input_image * scaled_mask, output_image * scaled_mask)

    def size_loss(self, probabilities):
        loss = 0
        for probability, feature_dimension in zip(probabilities, self.hparams.feature_dimensions):
            batch_size = probability.shape[0]
            loss = loss + torch.sum(probability * feature_dimension) / batch_size
        return loss

    def report_figures(self, batch, output):
        self._figure_reporter.report_figures(batch["image"], output[0][0], output[2], output[3], self.global_step)
