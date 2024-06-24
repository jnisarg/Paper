"""
# Filename: __init__.py                                                        #
# Project: backbones                                                           #
# Created Date: Monday, June 24th 2024, 11:17:04 am                            #
# Author: Nisarg Joshi                                                         #
# -----                                                                        #
# Last Modified: Mon Jun 24 2024                                               #
# Modified By: Nisarg Joshi                                                    #
# -----                                                                        #
# Copyright (c) 2024 Nisarg Joshi @ HL Klemove India Pvt. Ltd.                 #
"""

from .ddrnet import DDRNet

backbone_hub: dict = {"ddrnet": DDRNet}
