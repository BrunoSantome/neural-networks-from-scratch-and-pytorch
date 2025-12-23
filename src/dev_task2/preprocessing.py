import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


"""
Details 

Aerial Landscape Images 

(161.48 MB) upload to git. 

Image Resolution: 256x256 pixels
Number of categories: 15
Categories: 
Agriculture, Airport, Beach, City, Desert, 
Forest, Grassland,Highway, Lake, Mountain, 
Parking, Port, Railway, Residential, River

Number of Images per Category: 800

Sources: 
AID Dataset: https://captain-whu.github.io/AID/

NWPU-Resisc45 Dataset: https://paperswithcode.com/dataset/resisc45

"""

# PATH = "C:/Users/bruno/OneDrive/Escritorio/Desktop/Repositories/Programming_math_Ai_Assessmment_Msc_Ai/dataset/Aerial_Landscapes/"

PATH = "../../dataset/Aerial_Landscapes/"


def pre_processing_dataset(batch_size=32, train_test_split_n=0.2, seed=42, path=PATH):
    # Careful if adding augmentation or transformations to the data better to do for training,
    # so apply after the split not before.
    transform_train = transforms.Compose(
        [
            # transforms.Resize((256, 256)), ## check if reducing the size is worth it.
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            # transforms.Resize((256, 256)), ## check if reducing the size is worth it.
            transforms.ToTensor(),
        ]
    )
    landscapes = datasets.ImageFolder(root=path)
    training_split_n = 1 - train_test_split_n
    gen = torch.Generator().manual_seed(
        seed
    )  # generator to have a fixed seed when splitting. to recreate results.
    train_size = int(training_split_n * len(landscapes))
    train_set, test_set = random_split(
        landscapes, [train_size, len(landscapes) - train_size], generator=gen
    )
    train_set.dataset.transform = transform_train
    test_set.dataset.transform = transform_test
    # DataLoader: Groups samples into batches,
    # Shuffles data, loads data in parallel,
    # Feeds data efficiently to the GPU
    # The batching helps for a faster GPU computation and more stable gradients

    landscapes_train = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    landscapes_test = DataLoader(test_set, batch_size=batch_size, num_workers=2)

    # Disk >  Dataset > Transform > DataLoader > Model

    # for X, y in landscapes_test:
    #     print(f"Shape of X : {X.shape}")
    #     print(f"Shape of y: {y.shape} {y.dtype}")
    #     break
    return landscapes_train, landscapes_test
