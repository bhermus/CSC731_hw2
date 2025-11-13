from torchvision import datasets, transforms
from torch.utils.data import random_split


def load_mnist_dataset(root_dir="./data"):
    full_train_dataset = datasets.MNIST(
        root=root_dir,
        train=True,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=root_dir,
        train=False,
        download=True,
    )

    train_size = 50_000
    validation_size = 10_000

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size]
    )

    return train_dataset, test_dataset
