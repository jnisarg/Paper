"""
# Filename: head.py                                                            #
# Project: modules                                                             #
# Created Date: Monday, June 24th 2024, 9:17:05 am                             #
# Author: Nisarg Joshi                                                         #
# -----                                                                        #
# Last Modified: Mon Jun 24 2024                                               #
# Modified By: Nisarg Joshi                                                    #
# -----                                                                        #
# Copyright (c) 2024 Nisarg Joshi @ HL Klemove India Pvt. Ltd.                 #
"""

import torch
from torch import nn

from . import common as cm


class ClassificationHead(nn.Module):
    """
    Module for classification and segmentation head.

    Args:
        in_channels (int): Number of input channels.
        head_channels (int): Number of channels in the head.
        num_classes (int): Number of classes.

    Attributes:
        classifier (nn.Sequential): Sequential module for classification head.

    """

    def __init__(self, in_channels: int, head_channels: int, num_classes: int) -> None:
        """Initialize the module."""
        super().__init__()

        self.classifier = nn.Sequential(
            cm.BNReLUConv(in_channels, head_channels, kernel_size=3),
            cm.BNReLUConv(head_channels, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.classifier(x)


class RegressionHead(nn.Module):
    """
    Module for regression head.

    Args:
        in_channels (int): Number of input channels.
        head_channels (int): Number of channels in the head.

    Attributes:
        regressor (nn.Sequential): Sequential module for regression head.

    """

    def __init__(self, in_channels: int, head_channels: int) -> None:
        """Initialize the module."""
        super().__init__()

        self.regressor = nn.Sequential(
            cm.BNReLUConv(in_channels, head_channels, kernel_size=3),
            cm.BNReLUConv(head_channels, 4, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.regressor(x)
