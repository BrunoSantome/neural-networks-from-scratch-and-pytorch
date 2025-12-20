import torch
import torch.nn as nn
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from preprocessing import pre_processing_dataset
from torchvision import datasets, transforms
import sys


def swtich_to_cuda():
    print(sys.executable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    print(torch.cuda.get_device_name(0))

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)


PATH = "C:/Users/bruno/OneDrive/Escritorio/Desktop/Repositories/Programming_math_Ai_Assessmment_Msc_Ai/dataset/Aerial_Landscapes/"


# class NeuronalNetwork(nn.Module):
#     def __init__(self):
#         super().__init__


if __name__ == "__main__":
    X_train, y_train = pre_processing_dataset()
    # for X, y in X_train:
    #     print(f"Shape of X : {X.shape}")
    #     print(f"Shape of y: {y.shape} {y.dtype}")
    #     break
