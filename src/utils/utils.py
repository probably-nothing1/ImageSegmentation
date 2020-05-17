import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

from src.models import UNet

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(path))
    return model.eval()
