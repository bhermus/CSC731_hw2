from sympy.printing.pytorch import torch
from torch.utils.data import DataLoader

from cnn import MyCnn
from download_mnist import load_mnist_dataset


def main():
    train_dataset, test_dataset = load_mnist_dataset()
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    dummy_input = torch.randn(64, 1, 28, 28)

    model = MyCnn()
    output = model(dummy_input)

    print("✅ Model defined successfully.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape (logits for 10 classes): {output.shape}")

if __name__ == '__main__':
    main()
