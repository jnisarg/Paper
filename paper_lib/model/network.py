"""
# Filename: network.py                                                         #
# Project: model                                                               #
# Created Date: Monday, June 24th 2024, 9:16:56 am                             #
# Author: Nisarg Joshi                                                         #
# -----                                                                        #
# Last Modified: Mon Jun 24 2024                                               #
# Modified By: Nisarg Joshi                                                    #
# -----                                                                        #
# Copyright (c) 2024 Nisarg Joshi @ HL Klemove India Pvt. Ltd.                 #
"""

import torch
from torch import nn

from paper_lib.model.modules import common as cm
from paper_lib.model.backbones.ddrnet import DDRNet
from paper_lib.model.modules.head import ClassificationHead, RegressionHead


class Network(nn.Module):
    """
    Network module for pixel-level classification and object localization tasks.

    Args:
        localization_classes (int): Number of localization classes.
        classification_classes (int): Number of classification classes.
        head_channels (int): Number of channels in the head.
        ppm_block (str, optional): Name of the PPM block. Defaults to "dappm".
    """

    def __init__(
        self,
        localization_classes: int,
        classification_classes: int,
        head_channels: int,
        ppm_block: str = "dappm",
    ) -> None:
        """
        Initialize the module.
        """
        super().__init__()

        self.head_channels = head_channels
        self.localization_classes = localization_classes
        self.classification_classes = classification_classes

        # Backbone network
        self.backbone = DDRNet(ppm_block=ppm_block, planes=32, ppm_planes=128)

        # Heads
        self.classifier = ClassificationHead(
            in_channels=self.backbone.out_channels[0],
            head_channels=head_channels,
            num_classes=classification_classes,
        )
        self.regressor = RegressionHead(
            in_channels=self.backbone.out_channels[0],
            head_channels=head_channels,
        )

        # Initialize weights
        cm.init_weights(self)

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of output pixel-level classification
            and object localization tensors.
        """
        # Pass input through backbone network
        ppm, detail5, _ = self.backbone(input_tensor)

        # Pass output through classification and localization heads
        return self.classifier(ppm + detail5), self.regressor(ppm + detail5)


if __name__ == "__main__":
    model = Network(
        localization_classes=7,
        classification_classes=19,
        head_channels=32,
        ppm_block="dappm",
    )
    model.eval()
    print(model)

    x = torch.randn(1, 3, 1024, 1024)
    out = model(x)
    print(f"Classification output shape: {out[0].shape}, Localization output shape: {out[1].shape}")

    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {parameters / 1e6:.2f}M")  # 3.87M
