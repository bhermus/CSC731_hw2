import torch
from torch.utils.data import DataLoader

from cnn import MyCnn
from download_mnist import load_mnist_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 64


def main():
    train_dataset, test_dataset = load_mnist_dataset()
    batch_size = BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    dummy_input = torch.randn(64, 1, 28, 28)

    model = MyCnn().to(DEVICE)
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape (logits for 10 classes): {output.shape}")


if __name__ == '__main__':
    main()
