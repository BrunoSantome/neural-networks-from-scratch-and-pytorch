import torch
import torch.nn as nn
from torchmetrics import Accuracy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocessing import pre_processing_dataset
import sys
import time


def swtich_to_cuda():
    print(sys.executable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    print(torch.cuda.get_device_name(0))

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    return device


PATH = "C:/Users/bruno/OneDrive/Escritorio/Desktop/Repositories/Programming_math_Ai_Assessmment_Msc_Ai/dataset/Aerial_Landscapes/"


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        torch.manual_seed(42)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3 * 256 * 256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 15),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_network(model, optimizer, loss_function, num_epochs, X_train, device):
    loss_across_epochs = []
    time_start = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in X_train:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output_data = model(inputs)
            loss = loss_function(output_data, labels)
            # L1_loss = 0
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        num_samples = len(X_train.dataset)
        loss_across_epochs.append(train_loss / num_samples)
        print(f"Epoch: {epoch} - Loss: {train_loss / num_samples:.4f}")
        # if epoch % 10 == 0:
        #     print(f"Epoch: {epoch} - Loss: {train_loss / num_samples:.4f}")
    print("--- %s seconds ---" % (time.time() - time_start))

    return loss_across_epochs


def evaluate_model(model, test_set, losses, device):
    acc = Accuracy(task="multiclass", num_classes=15).to(device)

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_set:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            acc(preds, labels)

    test_accuracy = acc.compute().item()
    print(f"Test accuracy: {test_accuracy:.4f}")
    return test_accuracy

    # Compute total test accuracy
    # test_accuracy = acc.compute()
    # print(f"Test accuracy: {test_accuracy}")
    # plt.figure()
    # plt.plot(batch_accuracies)
    # plt.xlabel("Batch index")
    # plt.ylabel("Accuracy")
    # plt.title("Batch Accuracy on Test Set")
    # plt.ylim(0, 1)
    # plt.show()


def plot_losses(losses):
    # Plot the loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    X_train, y_test = pre_processing_dataset(path=PATH)
    device = swtich_to_cuda()
    model_NN = NeuralNetwork().to(device)
    adam_optimizer = torch.optim.Adam(
        model_NN.parameters(), betas=(0.9, 0.999), lr=0.001, eps=1e-8, weight_decay=0.0
    )
    # adamW_optimizer = torch.optim.AdamW(
    #     model_NN.parameters(), lr=3e-4, weight_decay=1e-4
    # )
    # Train model
    lossNN = train_network(
        model_NN, adam_optimizer, nn.CrossEntropyLoss(), 50, X_train, device
    )
    evaluate_model(model_NN, y_test, lossNN, device)
    plot_losses(lossNN)

    ###########
    # epoch 10. lr 0.001.
    # Epoch: 0 - Loss: 4.4644
    # Epoch: 1 - Loss: 2.1895
    # Epoch: 2 - Loss: 2.0818
    # Epoch: 3 - Loss: 2.0591
    # Epoch: 4 - Loss: 2.0141
    # Epoch: 5 - Loss: 1.9654
    # Epoch: 6 - Loss: 1.9349
    # Epoch: 7 - Loss: 1.9370
    # Epoch: 8 - Loss: 1.9365
    # Epoch: 9 - Loss: 1.9005
    # --- 609.2014908790588 seconds ---

    # lr=0.01
    #     Epoch: 0 - Loss: 24.0916
    # Epoch: 1 - Loss: 2.7370
    # Epoch: 2 - Loss: 2.7099
    # Epoch: 3 - Loss: 2.7102
    # Epoch: 4 - Loss: 2.7101
    # Epoch: 5 - Loss: 2.7100
    # for X, y in X_train:
    #     print(y.min(), y.max())
    #     print(y.dtype)
    #     # print(output_data.shape)
    #     print(f"Shape of X : {X.shape}")
    #     print(f"Shape of y: {y.shape} {y.dtype}")
    #     break


# --- 846.3008494377136 seconds --- without betas and eps
# Test accuracy: 0.3454166650772095 with 20 epoch.
# adam_optimizer = torch.optim.Adam(model_NN.parameters(), lr=0.001, weight_decay=0.0)


# --- 511.83825612068176 seconds --- With betas and eps. Same epochs, same model but we reduced training time 5.58 minutes.
# ~Test accuracy: 0.3454


####--- 1481.3784992694855 seconds --- 50 epoch. # A lot of images and large and regular neuronal network → too many parameters, slow learning.

# Test accuracy: 0.3442 #Accuracy has not improved
