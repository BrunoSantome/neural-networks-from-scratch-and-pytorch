import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import sys

print(sys.executable)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
print(torch.cuda.get_device_name(0))

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)


class NeuronalNetwork(nn.Module):
    def __init__(self):
        super().__init__
