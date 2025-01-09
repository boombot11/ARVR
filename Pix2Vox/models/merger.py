# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class Merger(torch.nn.Module):
  def __init__(self, cfg):
        super(Merger, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.LeakyReLU(cfg.NETWORK['LEAKY_VALUE'])
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(16, 8, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.LeakyReLU(cfg.NETWORK['LEAKY_VALUE'])
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 4, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(4),
            torch.nn.LeakyReLU(cfg.NETWORK['LEAKY_VALUE'])
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(4, 2, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(2),
            torch.nn.LeakyReLU(cfg.NETWORK['LEAKY_VALUE'])
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(2, 1, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.LeakyReLU(cfg.NETWORK['LEAKY_VALUE'])
        )

  def forward(self, raw_features, coarse_volumes):
    # Process raw_features without splitting
    volume_weights = self.layer1(raw_features)  # Output: [batch_size, 16, 32, 32, 32]
    volume_weights = self.layer2(volume_weights)  # Output: [batch_size, 8, 32, 32, 32]
    volume_weights = self.layer3(volume_weights)  # Output: [batch_size, 4, 32, 32, 32]
    volume_weights = self.layer4(volume_weights)  # Output: [batch_size, 2, 32, 32, 32]
    volume_weights = self.layer5(volume_weights)  # Output: [batch_size, 1, 32, 32, 32]

    volume_weights = torch.softmax(volume_weights, dim=1)  # Softmax along the channel dimension
    # Apply volume weights to coarse volumes
    coarse_volumes = coarse_volumes * volume_weights  # Element-wise multiplication
    coarse_volumes = torch.sum(coarse_volumes, dim=1)  # Sum along the channel dimension

    return torch.clamp(coarse_volumes, min=0, max=1)

