import math
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm


class UNetColorization(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(UNetColorization, self).__init__()

        def down_block(in_feat, out_feat, normalize=True):
            layers = [
                nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_feat, out_feat, dropout=0.0):
            layers = [
                nn.ConvTranspose2d(
                    in_feat, out_feat, kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.down1 = down_block(in_channels, 64, normalize=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)
        self.down6 = down_block(512, 512)
        self.down7 = down_block(512, 512)
        self.down8 = down_block(512, 512, normalize=False)

        self.up1 = up_block(512, 512, dropout=0.5)
        self.up2 = up_block(1024, 512, dropout=0.5)
        self.up3 = up_block(1024, 512, dropout=0.5)
        self.up4 = up_block(1024, 512)
        self.up5 = up_block(1024, 256)
        self.up6 = up_block(512, 128)
        self.up7 = up_block(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(
                128, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))

        return u8


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

model = UNetColorization()
loss_fn = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=3e-4)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

model.to(device)
# model(torch.randn(1, 1, 512, 512).to(device)) # Example input tensor with shape (batch_size, in_channels, height, width)

gray_path = "./open/train_mask/"
color_path = "./open/train_gt/"
img_names = os.listdir(gray_path)
train_len = len(img_names)

min_loss = 1
count = 0
BATCH_SIZE = 8
for epoch in tqdm(range(1, 10 + 1)):
    for i in tqdm(range(math.ceil(train_len / BATCH_SIZE)), leave=False):
        optimizer.zero_grad()  # 기울기 초기화

        gray_list = [transform(Image.open(gray_path + img).convert("L")).unsqueeze(0) for img in img_names[i:i+BATCH_SIZE]]
        color_list = [transform(Image.open(color_path + img).convert("RGB")).unsqueeze(0) for img in img_names[i:i+BATCH_SIZE]]

        gray = torch.cat(gray_list, dim=0).to(device)
        color = torch.cat(color_list, dim=0).to(device)

        pred = model(gray)
        loss = loss_fn(pred, color)
        # print(loss.item())
        loss.backward()
        optimizer.step()

        # print(f"\r[Epoch {epoch}/10] [Batch {i}/{train_len}] [loss: {loss.item()}]", end="")

        if min_loss > loss.item():
            min_loss = loss.item()
            torch.save(model.state_dict(), "./model/colorization.pth")
            # print(f"Model saved with loss: {min_loss}")
        count = count + 1
