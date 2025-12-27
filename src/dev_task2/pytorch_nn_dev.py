import torch
import torch.nn as nn
from torchmetrics import Accuracy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocessing import pre_processing_dataset
import sys
import time
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def swtich_to_cuda():
    print(sys.executable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    print(torch.cuda.get_device_name(0))

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    return device


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
        x = self.linear_relu_stack(x)
        return x


class CNN(nn.Module):
    # Large images + MLP → too many parameters, slow learning.
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.linear_cnn_model = nn.Sequential(  # The more out channels the more compute with bigger in channels
            nn.Conv2d(
                in_channels=in_channels, out_channels=32, kernel_size=3, padding=1
            ),  # Padding so input and output have same dimensions
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),  # We had to add another layer. otherwise too many parameters.
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifierNN = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(64 * 64 * 64, 512),
            # nn.Linear(256 * 16 * 16, 512),
            nn.Linear(256 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Adding dropout
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.linear_cnn_model(x)
        x = self.classifierNN(x)
        return x


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
    batch_accuracies = []
    with torch.no_grad():
        for inputs, labels in test_set:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            acc(preds, labels)

            batch_acc = (preds == labels).float().mean().item()
            batch_accuracies.append(batch_acc)

    test_accuracy = acc.compute().item()
    print(f"Test accuracy: {test_accuracy:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(batch_accuracies, marker="o")
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.title("Batch-wise Accuracy")
    plt.grid(True)
    plt.show()
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


def run_NN_model(device, X_train, y_test):
    ##############Neuronal network model###########33333
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


def run_CNN_model(device, X_train, y_test):
    ##############Convolutional Neuronal network model###############
    model_CNN = CNN(3, 15).to(device)
    optimizer_CNN = torch.optim.Adam(model_CNN.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    lossCNN = train_network(
        model_CNN, optimizer_CNN, loss_fn, num_epochs=10, X_train=X_train, device=device
    )
    evaluate_model(model_CNN, y_test, lossCNN, device)
    plot_losses(lossCNN)


if __name__ == "__main__":
    set_seed(42)
    X_train, y_test = pre_processing_dataset()
    device = swtich_to_cuda()

    ##############Neuronal network model###########33333
    # run_NN_model(device, X_train, y_test)

    ##############Convolutional Neuronal network model###############
    # run_CNN_model(device, X_train, y_test)
    model_CNN = CNN(3, 15).to(device)
    optimizer_CNN = torch.optim.Adam(model_CNN.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()  # Check this
    lossCNN = train_network(
        model_CNN,
        optimizer_CNN,
        loss_fn,
        num_epochs=50,
        X_train=X_train,
        device=device,
    )
    evaluate_model(model_CNN, y_test, lossCNN, device)
    plot_losses(lossCNN)

##################### NN TESTS #############3
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


# --- 846.3008494377136 seconds --- without betas and eps
# Test accuracy: 0.3454166650772095 with 20 epoch.
# adam_optimizer = torch.optim.Adam(model_NN.parameters(), lr=0.001, weight_decay=0.0)


# --- 511.83825612068176 seconds --- With betas and eps. Same epochs, same model but we reduced training time 5.58 minutes.
# ~Test accuracy: 0.3454


####--- 1481.3784992694855 seconds --- 50 epoch. # A lot of images and large and regular neuronal network → too many parameters, slow learning.

# Test accuracy: 0.3442 #Accuracy has not improved


############## CNN TESTS##############

# With only 1 epoch the loss is already lower compared to NN and accuracy increased drastically !
# Epoch: 0 - Loss: 1.7144
# --- 40.928863525390625 seconds ---
# Test accuracy: 0.5496

### 10 epochs 2 Conv layers
# Epoch: 2 - Loss: 0.8572
# Epoch: 3 - Loss: 0.4840
# Epoch: 4 - Loss: 0.2063
# Epoch: 5 - Loss: 0.0996
# Epoch: 6 - Loss: 0.0480
# Epoch: 7 - Loss: 0.0269
# Epoch: 8 - Loss: 0.0585
# Epoch: 9 - Loss: 0.0482
# --- 467.5640649795532 seconds ---
# Test accuracy: 0.6012

### 10 epochs 3 Conv layers
# Epoch: 5 - Loss: 0.2072
# Epoch: 6 - Loss: 0.1184
# Epoch: 7 - Loss: 0.0696
# Epoch: 8 - Loss: 0.0756
# Epoch: 9 - Loss: 0.0697
# --- 419.09427309036255 seconds ---
# Test accuracy: 0.6704

# Almost same time, but higher accuracy

### 10 epochs 4 Conv layers
# Epoch: 0 - Loss: 1.8732
# Epoch: 1 - Loss: 1.2456
# Epoch: 2 - Loss: 0.9269
# Epoch: 3 - Loss: 0.7283
# Epoch: 4 - Loss: 0.5444
# Epoch: 5 - Loss: 0.3610
# Epoch: 6 - Loss: 0.2367
# Epoch: 7 - Loss: 0.1363
# Epoch: 8 - Loss: 0.1407
# Epoch: 9 - Loss: 0.0846
# --- 452.347186088562 seconds ---
# Test accuracy: 0.7125

# learning rate 0.002 and 30 epoch nope.
# Epoch: 0 - Loss: 2.0310
# Epoch: 1 - Loss: 1.2746
# Epoch: 2 - Loss: 0.9880
# Epoch: 3 - Loss: 0.7867
# Epoch: 4 - Loss: 0.6283
# Epoch: 5 - Loss: 0.4735
# Epoch: 6 - Loss: 0.3513
# Epoch: 7 - Loss: 0.2778
# Epoch: 8 - Loss: 0.2037
# Epoch: 9 - Loss: 0.1288
# Epoch: 10 - Loss: 0.0956
# Epoch: 11 - Loss: 0.1343
# Epoch: 12 - Loss: 0.0898
# Epoch: 13 - Loss: 0.0857
# Epoch: 14 - Loss: 0.0493
# Epoch: 15 - Loss: 0.0840
# Epoch: 16 - Loss: 0.0853
# Epoch: 17 - Loss: 0.0559
# Epoch: 18 - Loss: 0.0779
# Epoch: 19 - Loss: 0.0292
# Epoch: 20 - Loss: 0.0647
# Epoch: 21 - Loss: 0.0473
# Epoch: 22 - Loss: 0.0146
# Epoch: 23 - Loss: 0.0570
# Epoch: 24 - Loss: 0.0459
# Epoch: 25 - Loss: 0.0703
# Epoch: 26 - Loss: 0.0383
# Epoch: 27 - Loss: 0.0793
# Epoch: 28 - Loss: 0.0273
# Epoch: 29 - Loss: 0.0303
# --- 1282.694759130478 seconds ---
# Test accuracy: 0.7133

### 20 epochs 4 Conv layers 0.001
# Epoch: 0 - Loss: 1.7709
# Epoch: 1 - Loss: 1.1802
# Epoch: 2 - Loss: 0.9443
# Epoch: 3 - Loss: 0.7702
# Epoch: 4 - Loss: 0.6457
# Epoch: 5 - Loss: 0.4734
# Epoch: 6 - Loss: 0.3574
# Epoch: 7 - Loss: 0.2489
# Epoch: 8 - Loss: 0.1515
# Epoch: 9 - Loss: 0.1465
# Epoch: 10 - Loss: 0.0975
# Epoch: 11 - Loss: 0.0812
# Epoch: 12 - Loss: 0.0801
# Epoch: 13 - Loss: 0.0974
# Epoch: 14 - Loss: 0.0574
# Epoch: 15 - Loss: 0.0719
# Epoch: 16 - Loss: 0.0457
# Epoch: 17 - Loss: 0.0200
# Epoch: 18 - Loss: 0.0882
# Epoch: 19 - Loss: 0.0445
# --- 775.4758677482605 seconds ---
# Test accuracy: 0.7175

### 10 epochs 5 Conv layers
# Epoch: 0 - Loss: 1.9189
# Epoch: 1 - Loss: 1.2982
# Epoch: 2 - Loss: 1.0630
# Epoch: 3 - Loss: 0.8757
# Epoch: 4 - Loss: 0.7186
# Epoch: 5 - Loss: 0.5930
# Epoch: 6 - Loss: 0.4173
# Epoch: 7 - Loss: 0.3264
# Epoch: 8 - Loss: 0.2364
# Epoch: 9 - Loss: 0.2000
# --- 817.5077738761902 seconds ---
# Test accuracy: 0.7433


# Epoch: 0 - Loss: 2.0397
# Epoch: 1 - Loss: 1.3315
# Epoch: 2 - Loss: 1.0799
# Epoch: 3 - Loss: 0.9045
# Epoch: 4 - Loss: 0.7880
# Epoch: 5 - Loss: 0.6868
# Epoch: 6 - Loss: 0.5774
# Epoch: 7 - Loss: 0.4853
# Epoch: 8 - Loss: 0.4058
# Epoch: 9 - Loss: 0.3124
# --- 409.35135293006897 seconds ---
# Test accuracy: 0.7563

# 20 epoch same hyperaparameters, to see if more epochs improves test accuracy.
# more epoch means lower loss but does not mean lower accuracy, it is actually lower.
# Epoch: 12 - Loss: 0.1406
# Epoch: 13 - Loss: 0.0868
# Epoch: 14 - Loss: 0.0876
# Epoch: 15 - Loss: 0.0768
# Epoch: 16 - Loss: 0.0765
# Epoch: 17 - Loss: 0.0938
# Epoch: 18 - Loss: 0.0731
# Epoch: 19 - Loss: 0.1201
# --- 840.0892059803009 seconds ---
# Test accuracy: 0.7038

# Same hyperparameters with 5 epochs.
# CUDA version: 12.8
# Epoch: 0 - Loss: 2.0451
# Epoch: 1 - Loss: 1.2884
# Epoch: 2 - Loss: 1.0574
# Epoch: 3 - Loss: 0.8840
# Epoch: 4 - Loss: 0.7546
# --- 213.03730249404907 seconds ---
# Test accuracy: 0.7333

# Adding dropout to the fully conected layer increase accuracy to 0.7779
# I will try now more epochs.
# Epoch: 0 - Loss: 2.3168
# Epoch: 1 - Loss: 1.6249
# Epoch: 2 - Loss: 1.2758
# Epoch: 3 - Loss: 1.1004
# Epoch: 4 - Loss: 0.9684
# Epoch: 5 - Loss: 0.8911
# Epoch: 6 - Loss: 0.7790
# Epoch: 7 - Loss: 0.7042
# Epoch: 8 - Loss: 0.6427
# Epoch: 9 - Loss: 0.5621
# --- 410.8563423156738 seconds ---
# Test accuracy: 0.7779


# CUDA version: 12.8
# Epoch: 0 - Loss: 2.1905
# Epoch: 1 - Loss: 1.6136
# Epoch: 2 - Loss: 1.3428
# Epoch: 3 - Loss: 1.1009
# Epoch: 4 - Loss: 0.9268
# Epoch: 5 - Loss: 0.8178
# Epoch: 6 - Loss: 0.7101
# Epoch: 7 - Loss: 0.6288
# Epoch: 8 - Loss: 0.5207
# Epoch: 9 - Loss: 0.4792
# Epoch: 10 - Loss: 0.3860
# Epoch: 11 - Loss: 0.3213
# Epoch: 12 - Loss: 0.2738
# Epoch: 13 - Loss: 0.2250
# Epoch: 14 - Loss: 0.2609
# Epoch: 15 - Loss: 0.1870
# Epoch: 16 - Loss: 0.1966
# Epoch: 17 - Loss: 0.1954
# Epoch: 18 - Loss: 0.1465
# Epoch: 19 - Loss: 0.1294
# Epoch: 20 - Loss: 0.1404
# Epoch: 21 - Loss: 0.1421
# Epoch: 22 - Loss: 0.1444
# Epoch: 23 - Loss: 0.1211
# Epoch: 24 - Loss: 0.1197
# Epoch: 25 - Loss: 0.0878
# Epoch: 26 - Loss: 0.1496
# Epoch: 27 - Loss: 0.0902
# Epoch: 28 - Loss: 0.0964
# Epoch: 29 - Loss: 0.1031
# --- 1224.7226314544678 seconds ---
# Test accuracy: 0.7821


# Final model, could be improved but it is enough maybe try with further epochs.
# Adaptativepooling as last layer of CNN,
# Epoch: 47 - Loss: 0.1203
# Epoch: 48 - Loss: 0.0810
# Epoch: 49 - Loss: 0.0845
# --- 1883.2752048969269 seconds ---
# Test accuracy: 0.8929

# Epoch: 55 - Loss: 0.0701
# Epoch: 56 - Loss: 0.0847
# Epoch: 57 - Loss: 0.0710
# Epoch: 58 - Loss: 0.0492
# Epoch: 59 - Loss: 0.0517
# --- 2579.0179855823517 seconds ---
# Test accuracy: 0.8763

# 4 Convolutional networks, dropout and Adaptativepooling
# Epoch: 43 - Loss: 0.2315
# Epoch: 44 - Loss: 0.2306
# Epoch: 45 - Loss: 0.2383
# Epoch: 46 - Loss: 0.1938
# Epoch: 47 - Loss: 0.1907
# Epoch: 48 - Loss: 0.2019
# Epoch: 49 - Loss: 0.1770
# --- 2126.95081448555 seconds ---
# Test accuracy: 0.8808
