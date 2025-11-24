import argparse
import torch
from torch.utils.data import DataLoader

from cnn import MyCnn
from download_mnist import load_mnist_dataset
from lstm import MyLstm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 64


def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += len(labels)
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, required=True, choices=["cnn", "lstm"],
                        help="Choose a network, either CNN or LSTM")
    args = parser.parse_args()
    print(f"Running using {args.network}")

    train_dataset, test_dataset, validation_dataset = load_mnist_dataset()
    batch_size = BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    model = None
    if args.network.lower() == "cnn":
        model = MyCnn().to(DEVICE)
    elif args.network.lower() == "lstm":
        model = MyLstm().to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_loss, accuracy = validate(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Loss: {avg_loss}, Val Loss: {val_loss}, Accuracy: {accuracy}")

    print("Final test stats:")
    test_loss, test_accuracy = validate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == '__main__':
    main()
