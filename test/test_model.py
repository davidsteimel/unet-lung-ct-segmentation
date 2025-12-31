import pytest
import torch
from unet.unet_model import UNet


def test_unet_output():
    model = UNet(n_channels=1, n_classes=1)
    dummy_input = torch.randn(1, 1, 160, 160)
    output = model(dummy_input)
    assert output.shape == dummy_input.shape, \
    f"Error: Input size {dummy_input.shape} does not match output size {output.shape} - size was changed"


def test_unet_odd_input_size():
    model = UNet(n_channels=1, n_classes=1)
    dummy_input = torch.randn(1, 1, 161, 161) 
    output = model(dummy_input)
    assert output.shape == dummy_input.shape, \
    f"Padding error: Input size {dummy_input.shape} does not match output size {output.shape}"