import torch
from torch import nn
from torch.utils.data import DataLoader
from homework.datasets.road_dataset import load_data
from homework.models import Detector, calculate_model_size_mb
from homework.models import save_model


def train():
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using mps")
    else:
        device = torch.device("cpu")
        print("Using cpu")

    # Configurations
    train_dataset_path = "../road_data/train"
    validation_dataset_path = "../road_data/val"
    num_epochs = 30
    batch_size = 128
    learning_rate = 1e-4

    # Load training and validation data
    train_loader = load_data(train_dataset_path, transform_pipeline="aug", return_dataloader=True, batch_size=batch_size, shuffle=True)
    val_loader = load_data(validation_dataset_path, transform_pipeline="default", return_dataloader=True, batch_size=batch_size, shuffle=False)

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")

    # Initialize model, optimizer, and loss functions
    model = Detector().to(device)
    print(model)

    # print model size
    model_size_mb = calculate_model_size_mb(model)
    print(f"Model size: {model_size_mb:.2f} MB")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    segmentation_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.L1Loss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_segmentation_loss = 0.0
        train_depth_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            seg_targets = batch["track"].to(device)  # Ground truth segmentation labels
            depth_targets = batch["depth"].to(device)  # Ground truth depth

            optimizer.zero_grad()  # Zero gradients

            # Forward pass
            logits, raw_depth = model(images)

            # Compute losses
            segmentation_loss = segmentation_loss_fn(logits, seg_targets)
            depth_loss = depth_loss_fn(raw_depth, depth_targets)

            # Total loss
            total_loss = segmentation_loss + depth_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            # Accumulate training loss
            train_segmentation_loss += segmentation_loss.item()
            train_depth_loss += depth_loss.item()

        # Calculate average losses per epoch
        avg_train_segmentation_loss = train_segmentation_loss / len(train_loader)
        avg_train_depth_loss = train_depth_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_segmentation_loss = 0.0
        val_depth_loss = 0.0

        with torch.no_grad():  # No gradient computation during validation
            for batch in val_loader:
                images = batch["image"].to(device)
                seg_targets = batch["track"].to(device)
                depth_targets = batch["depth"].to(device)

                logits, raw_depth = model(images)

                # Compute validation losses
                val_segmentation_loss += segmentation_loss_fn(logits, seg_targets).item()
                val_depth_loss += depth_loss_fn(raw_depth, depth_targets).item()


        avg_val_segmentation_loss = val_segmentation_loss / len(val_loader)
        avg_val_depth_loss = val_depth_loss / len(val_loader)

        # Print losses for the epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}]: "
              f"Train Segmentation Loss: {avg_train_segmentation_loss:.4f}, "
              f"Train Depth Loss: {avg_train_depth_loss:.4f}, "
              f"Val Segmentation Loss: {avg_val_segmentation_loss:.4f}, "
              f"Val Depth Loss: {avg_val_depth_loss:.4f}")


    # Save the model
    save_model(model)

if __name__ == "__main__":
    train()
