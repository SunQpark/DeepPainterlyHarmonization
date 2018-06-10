import sys, os
import numpy as np
import torch
import torch.nn as nn


def select_mask(features, mask=None):
    """
    select activation vectors in mask
    features: (c, w, h)
    mask: (w, h)
    """
    c, w, h = features.shape
    if mask is not None:
        activations = torch.masked_select(features[i], mask).view(c, -1)
    else:
        activations = features.view(c, -1)
    
    return torch.cat(activations, dim=0)