from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # Input Convolutional layer to increase channels
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual block 1
        self.res_block1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
        )

        # Convolutional layer with reduced channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Residual block 2
        self.res_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
        )

        # Convolutional layer with reduced channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Residual block 3
        self.res_block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
        )

        # Global Average Pooling and Final Classifier Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # Normalize the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # First convolution and residual block
        x = F.relu(self.bn1(self.conv1(z)))
        residual = x
        x = self.res_block1(x) + residual  # Residual connection 1

        # Second convolution and residual block
        x = F.relu(self.bn2(self.conv2(x)))
        residual = x
        x = self.res_block2(x) + residual  # Residual connection 2

        # Third convolution and residual block
        x = F.relu(self.bn3(self.conv3(x)))
        residual = x
        x = self.res_block3(x) + residual  # Residual connection 3

        # Global Average Pooling and Classifier
        x = self.global_avg_pool(x)
        logits = self.fc(x).squeeze()

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


class Detector(torch.nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            # If input and output channels differ, use 1x1 convolution to match dimensions
            self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) \
                if in_channels != out_channels else nn.Identity()

        def forward(self, x):
            identity = self.residual_connection(x)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += identity
            return F.relu(out)

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # TODO: implement
        # Encoder with additional residual block
        self.enc1 = self.ResidualBlock(in_channels, 32)          # No downsampling here
        self.enc2 = self.ResidualBlock(32, 64, stride=2)         # Downsampling 1
        self.enc3 = self.ResidualBlock(64, 128, stride=2)        # Downsampling 2
        self.enc4 = self.ResidualBlock(128, 256, stride=2)       # New residual block

        # Decoder for segmentation
        self.up1_seg = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsample 1
        self.up2_seg = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)   # Upsample 2
        self.up3_seg = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)    # New upsample 3
        self.final_seg = nn.Conv2d(32, num_classes, kernel_size=1)

        # Decoder for depth estimation
        self.up1_depth = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1) # Upsample 1
        self.up2_depth = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # Upsample 2
        self.up3_depth = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)   # New upsample 3
        self.final_depth = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        # Encoder
        enc1 = self.enc1(z)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)  # New block

        # Segmentation decoder
        seg_up1 = self.up1_seg(enc4)
        seg_up2 = self.up2_seg(seg_up1 + enc3)  # Skip connection
        seg_up3 = self.up3_seg(seg_up2 + enc2)  # Skip connection
        logits = self.final_seg(seg_up3 + enc1)  # Skip connection

        # Depth decoder
        depth_up1 = self.up1_depth(enc4)
        depth_up2 = self.up2_depth(depth_up1 + enc3)  # Skip connection
        depth_up3 = self.up3_depth(depth_up2 + enc2)  # Skip connection
        raw_depth = torch.sigmoid(self.final_depth(depth_up3 + enc1)).squeeze(1)  # Skip connection

        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        # Post-process depth with min-max normalization
        # min_depth = raw_depth.amin(dim=[1, 2], keepdim=True)  # find min for each image in batch
        # max_depth = raw_depth.amax(dim=[1, 2], keepdim=True)  # find max for each image in batch
        # depth = (raw_depth - min_depth) / (max_depth - min_depth + 1e-5)  # avoid division by zero

        return pred, raw_depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
