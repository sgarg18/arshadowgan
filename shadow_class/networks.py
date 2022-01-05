# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class Generator_with_Refin(nn.Module):
    def __init__(self, encoder):
        """Generator initialization

        Args:
            encoder: an encoder for Unet generator
        """
        super(Generator_with_Refin, self).__init__()

        # declare Unet generator
        self.generator = smp.Unet(
            encoder_name=encoder,
            classes=1,
            activation='identity',
            encoder_depth=4,
            decoder_channels=[128, 64, 32, 16],
        )
        # replace the first conv block in generator (6 channels tensor as input)
        self.generator.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False)
        self.generator.segmentation_head = nn.Identity()

        # RGB-shadow mask as output before refinement module
        self.SG_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)

        # refinement module
        self.refinement = torch.nn.Sequential()
        for i in range(4):
            self.refinement.add_module(f'refinement{3*i+1}', nn.BatchNorm2d(16))
            self.refinement.add_module(f'refinement{3*i+2}', nn.ReLU())
            self.refinement.add_module(f'refinement{3*i+3}', nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1))

        # RGB-shadow mask as output after refinement module
        self.output1 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """Forward for generator

        Args:
            x: torch.FloatTensor or torch.cuda.FloatTensor - input tensor with images and masks
        """
        x = self.generator(x)
        out1 = self.SG_head(x)

        x = self.refinement(x)
        x = self.output1(x)
        return out1, x


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        """Discriminator initialization

        Args:
            input_shape (tuple): shape of input image
        """
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=4, stride=2, padding=1)) #k=3,p=1
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        """Discriminator forward
        """
        return self.model(img)


