# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch

class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Adding a Linear layer to adapt input size to match expected decoder input.
        # This is necessary because the input tensor size does not directly match the expected size.
        self.linear = torch.nn.Linear(20736, 2048 * 2 * 2 * 2)  # Adjust size here

        # Layer Definitions (keep as is, as we cannot change the original network structure)
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(2048, 512, kernel_size=4, stride=2, bias=cfg.NETWORK['TCONV_USE_BIAS'], padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=cfg.NETWORK['TCONV_USE_BIAS'], padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=cfg.NETWORK['TCONV_USE_BIAS'], padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.NETWORK['TCONV_USE_BIAS'], padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK['TCONV_USE_BIAS']),
            torch.nn.Sigmoid()
        )

    def forward(self, image_features):
        print(f"Decoder input shape: {image_features.shape}")
        
        # Flatten the input tensor
        flattened_size = image_features.numel()
        print(f"Flattened input size: {flattened_size}")
        
        # Check if the flattened size matches what we expect
        if flattened_size != 20736:
            raise ValueError(f"Unexpected flattened size: {flattened_size}.")
        
        # Use the Linear layer to adjust the size to match the decoder's input shape
        image_features = image_features.view(-1)  # Flatten the tensor to 1D
        image_features = self.linear(image_features)  # Adjust it to match the required size
        
        # Now reshape the output of the Linear layer to the expected shape for the decoder
        image_features = image_features.view(1, 2048, 2, 2, 2)  # Reshape to match the decoder's expected input size
        print(f"Reshaped image_features to {image_features.shape}")

        # Proceed with the rest of the layers
        gen_volume = self.layer1(image_features)
        gen_volume = self.layer2(gen_volume)
        gen_volume = self.layer3(gen_volume)
        gen_volume = self.layer4(gen_volume)
        
        raw_feature = gen_volume  # Store raw feature before the last layer
        gen_volume = self.layer5(gen_volume)
        raw_feature = torch.cat((raw_feature, gen_volume), dim=1)

        return raw_feature, gen_volume