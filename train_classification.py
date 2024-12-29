import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch import nn
from torch.utils.data.dataloader import DataLoader

from grader.datasets.classification_dataset import load_data
from homework.models import save_model, Classifier

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]



def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using mps")
    else:
        device = torch.device("cpu")
        print("Using cpu")

    # set random seed for reproducibility
    seed = 46
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = 128
    train_data = load_data("../classification_data/train", shuffle=True, batch_size=batch_size, num_workers=4, transform_pipeline="aug")
    val_data = load_data("../classification_data/val", shuffle=False, transform_pipeline="default")

    model = Classifier(num_classes=6).to(device)
    print(model)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = loss_func(logits, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        train_correct, train_total = 0, 0
        with torch.no_grad():
            for img, label in train_data:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                _, predicted = torch.max(logits, 1)
                train_total += label.size(0)
                train_correct += (predicted == label).sum().item()
        train_accuracy = train_correct / train_total

        val_correct, val_total = 0, 0
        with torch.no_grad():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                logits = model(img)
                _, predicted = torch.max(logits, 1)
                val_total += label.size(0)
                val_correct += (predicted == label).sum().item()
        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_data):.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        # stop training when validation accuracy reaches 0.92
        if val_accuracy >= 0.92:
            print("Validation accuracy reached 0.94, stopping training.")
            break

    # Save the model
    save_model(model)

if __name__ == "__main__":
    train()
