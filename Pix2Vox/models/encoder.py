# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models

class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 2048, kernel_size=3),  # This increases channels to 2048
            torch.nn.BatchNorm2d(2048),
            torch.nn.ELU(),
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(2048, 2048, kernel_size=3, stride=2),  # Downsample to [2048, 5, 5]
            torch.nn.BatchNorm2d(2048),
            torch.nn.ELU(),
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(2048, 2048, kernel_size=3, stride=2),  # Further downsample to [2048, 2, 2]
            torch.nn.BatchNorm2d(2048),
            torch.nn.ELU(),
        )

        # Add upsampling layer to match the decoder input shape
        self.upsample = torch.nn.ConvTranspose2d(2048, 2048, kernel_size=4, stride=2, padding=1)

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False

    def forward(self, rendering_images):
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.vgg(img.squeeze(dim=0))
            features = self.layer1(features)
            features = self.layer2(features)
            features = self.layer3(features)
            features = self.layer4(features)
            features = self.layer5(features)  # Downsample to [2048, 5, 5]
            features = self.layer6(features)  # Further downsample to [2048, 2, 2]
            features = self.upsample(features)  # Upsample to [2048, 4, 4] or [2048, 2, 2]
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        return image_features
