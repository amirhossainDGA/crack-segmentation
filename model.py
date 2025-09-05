import torch
import torch.nn as nn
import torch.nn.functional as F


class CrackSegNet(nn.Module):
    def __init__(self):
        super(CrackSegNet, self).__init__()

        # ---- Encoder (Downsampling) ----
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 512 -> 256
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 256 -> 128
        )

        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # ---- Decoder (Upsampling) ----
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 128 -> 256
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 256 -> 512
            nn.ReLU()
        )

        # ---- Output layer ----
        self.out = nn.Conv2d(16, 2, kernel_size=1)  
        # 2 classes: 0 = background, 1 = crack

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec1(x)
        x = self.dec2(x)

        # Output
        x = self.out(x)  # no activation (use CrossEntropyLoss later)
        return x


# Quick test
if __name__ == "__main__":
    model = CrackSegNet()
    dummy_input = torch.randn(1, 3, 512, 512)  # 1 image, 3 channels, 512x512
    output = model(dummy_input)
    print("Output shape:", output.shape)  # should be [1, 2, 512, 512]
