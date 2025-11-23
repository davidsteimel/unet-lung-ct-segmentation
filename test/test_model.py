import pytest
import torch
from unet.unet_model import UNet


def test_unet_output():
    model = UNet(n_channels=1, n_classes=1)
    dummy_input = torch.randn(1, 1, 160, 160)
    output = model(dummy_input)
    assert output.shape == dummy_input.shape, \
    f"Fehler: Input-Größe {dummy_input.shape} passt nicht zu Output-Größe {output.shape} - Größe wurde verändert"


def test_unet_odd_input_size():
    model = UNet(n_channels=1, n_classes=1)
    dummy_input = torch.randn(1, 1, 161, 161) 
    output = model(dummy_input)
    assert output.shape == dummy_input.shape, \
    f"Padding-Fehler: Input-Größe {dummy_input.shape} passt nicht zu Output-Größe {output.shape}"